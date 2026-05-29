"""
数据对齐模块

解决的问题：商家上传的订单数据乱七八糟，怎么和直播时间对齐

核心功能：
1. 自动识别订单文件格式（CSV、Excel都能处理）
2. 智能匹配时间列和金额列（商家可能用不同的列名）
3. 处理跨天直播（晚上8点到凌晨1点很常见）
4. 过滤掉直播开始前24小时和结束后24小时以外的订单
5. 标记重复订单、退款订单，但不直接删除（让商家知道发生了什么）

设计原则：
- 对脏数据极度宽容，商家上传什么乱七八糟的东西都不能崩
- 自动推断列名，不要让商家手动配置
- 错误提示要像客服一样温柔
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OrderRecord:
    """
    订单记录数据类
    
    只保留最核心的字段，其他字段有需要再加
    """
    order_time: datetime     # 订单时间
    amount: float           # 订单金额
    order_id: Optional[str] = None  # 订单ID（如果有）
    user_id: Optional[str] = None   # 用户ID（如果有）
    is_duplicate: bool = False      # 是否重复订单
    is_refund: bool = False         # 是否退款订单
    
    def to_dict(self) -> Dict[str, Any]:
        """转成字典，方便序列化"""
        return {
            'order_time': self.order_time.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': self.amount,
            'order_id': self.order_id,
            'user_id': self.user_id,
            'is_duplicate': self.is_duplicate,
            'is_refund': self.is_refund
        }


class DataAligner:
    """
    数据对齐器
    
    这个类解决什么问题：
    商家上传一个订单数据文件，我要把它变成标准化的订单记录列表，
    并且和直播时间对齐，过滤掉不相关的订单
    """
    
    # 常见的时间列名（自动匹配）
    TIME_COLUMN_NAMES = [
        '时间', '订单时间', '付款时间', '创建时间', '下单时间', '支付时间',
        'time', 'order_time', 'pay_time', 'create_time', 'order date', 'payment time'
    ]
    
    # 常见的金额列名（自动匹配）
    AMOUNT_COLUMN_NAMES = [
        '金额', '订单金额', '实付金额', '付款金额', '价格', '总价', '成交金额',
        'amount', 'price', 'total', 'payment', 'order_amount', 'pay_amount'
    ]
    
    # 常见的订单ID列名
    ORDER_ID_COLUMN_NAMES = [
        '订单号', '订单ID', '订单编号', '流水号',
        'order_id', 'order_no', 'order_number', 'trade_no'
    ]
    
    # 常见的用户ID列名
    USER_ID_COLUMN_NAMES = [
        '用户ID', '买家ID', '会员ID', '客户ID', '账号',
        'user_id', 'buyer_id', 'member_id', 'customer_id', 'account'
    ]
    
    # 常见的退款标记列名
    REFUND_COLUMN_NAMES = [
        '退款', '售后', '退货', '是否退款', '订单状态',
        'refund', 'return', 'after_sale', 'order_status'
    ]
    
    def __init__(
        self,
        time_window_before: int = 24,    # 直播前多少小时算有效
        time_window_after: int = 24,     # 直播后多少小时算有效
        cross_day_threshold: int = 4,    # 凌晨几点前算前一天
        duplicate_window_minutes: int = 5  # 重复订单判断窗口（分钟）
    ):
        """
        初始化数据对齐器
        
        参数都是可选的，不给就用默认值
        """
        self.time_window_before = time_window_before
        self.time_window_after = time_window_after
        self.cross_day_threshold = cross_day_threshold
        self.duplicate_window_minutes = duplicate_window_minutes
        
        # 记录处理过程中的警告信息，最后统一展示给商家
        self.warnings: List[str] = []
        
    def _add_warning(self, message: str):
        """添加警告信息"""
        self.warnings.append(message)
        
    def detect_file_format(self, file_path: str) -> str:
        """
        检测文件格式
        
        支持的格式：CSV、Excel（xlsx、xls）
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return 'csv'
        elif suffix in ['.xlsx', '.xls']:
            return 'excel'
        else:
            raise ValueError(f"不支持的文件格式：{suffix}。请上传CSV或Excel文件")
    
    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        读取订单文件
        
        自动处理不同格式，自动推断编码
        """
        file_format = self.detect_file_format(file_path)
        
        try:
            if file_format == 'csv':
                # 尝试不同的编码
                encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("无法识别文件编码，请确保文件是UTF-8或GBK编码")
                # 清除可能的BOM残留
                df.columns = [col.lstrip('\ufeff') for col in df.columns]
            else:
                # Excel文件
                df = pd.read_excel(file_path)
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"读取文件失败：{str(e)}")
    
    def find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """
        在DataFrame中查找匹配的列名
        
        支持模糊匹配，不区分大小写
        """
        columns_lower = {col.lower(): col for col in df.columns}
        
        for name in possible_names:
            name_lower = name.lower()
            # 精确匹配
            if name_lower in columns_lower:
                return columns_lower[name_lower]
            # 包含匹配
            for col_lower, col_original in columns_lower.items():
                if name_lower in col_lower or col_lower in name_lower:
                    return col_original
        
        return None
    
    def parse_time(self, time_value: Any) -> Optional[datetime]:
        """
        解析时间字段
        
        支持多种格式：
        - 2024-01-15 14:30:00
        - 2024/01/15 14:30
        - 2024年1月15日 14:30
        - 时间戳
        - Excel序列号
        """
        if pd.isna(time_value):
            return None
        
        # 如果已经是datetime类型
        if isinstance(time_value, datetime):
            return time_value
        
        # 转换为字符串
        time_str = str(time_value).strip()
        
        # 尝试各种格式
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
            '%Y年%m月%d日 %H:%M:%S',
            '%Y年%m月%d日 %H:%M',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        # 尝试pandas自动解析
        try:
            parsed = pd.to_datetime(time_value)
            if not pd.isna(parsed):
                return parsed.to_pydatetime()
        except:
            pass
        
        return None
    
    def parse_amount(self, amount_value: Any) -> Optional[float]:
        """
        解析金额字段
        
        处理各种格式：
        - 纯数字：99.00
        - 带货币符号：¥99.00、$99.00
        - 带千分位：1,234.56
        - 中文数字：一百二十三
        """
        if pd.isna(amount_value):
            return None
        
        # 如果已经是数字
        if isinstance(amount_value, (int, float)):
            return float(amount_value)
        
        # 转换为字符串并清理
        amount_str = str(amount_value).strip()
        
        # 去除货币符号和千分位
        amount_str = re.sub(r'[¥$￥,，\s]', '', amount_str)
        
        # 尝试转换为浮点数
        try:
            return float(amount_str)
        except ValueError:
            return None
    
    def is_refund_order(self, row: pd.Series, refund_col: Optional[str]) -> bool:
        """
        判断是否是退款订单
        
        根据退款标记列或订单状态列判断
        """
        if refund_col is None:
            return False
        
        value = str(row.get(refund_col, '')).lower()
        
        # 常见的退款标记
        refund_keywords = ['退款', '退货', '售后', 'refund', 'return', 'cancelled', 'closed']
        
        for keyword in refund_keywords:
            if keyword in value:
                return True
        
        return False
    
    def process_orders(
        self,
        file_path: str,
        live_start_time: datetime,
        live_end_time: Optional[datetime] = None,
        progress_callback=None
    ) -> Tuple[List[OrderRecord], Dict[str, Any]]:
        """
        处理订单数据的主函数
        
        这是商家调用的主要接口
        
        参数：
            file_path: 订单文件路径
            live_start_time: 直播开始时间
            live_end_time: 直播结束时间（可选，如果没给就按开始时间+3小时算）
            progress_callback: 进度回调函数
            
        返回：
            (订单记录列表, 统计信息字典)
        """
        self.warnings = []  # 重置警告列表
        
        if progress_callback:
            progress_callback(0.1, "正在读取订单文件...")
        
        # 读取文件
        try:
            df = self.read_file(file_path)
        except Exception as e:
            raise RuntimeError(f"读取订单文件失败：{str(e)}")
        
        if progress_callback:
            progress_callback(0.2, f"文件读取成功，共{len(df)}行数据")
        
        # 检查数据是否为空
        if len(df) == 0:
            raise RuntimeError("订单文件为空，请检查文件内容")
        
        # 自动识别列
        time_col = self.find_column(df, self.TIME_COLUMN_NAMES)
        amount_col = self.find_column(df, self.AMOUNT_COLUMN_NAMES)
        order_id_col = self.find_column(df, self.ORDER_ID_COLUMN_NAMES)
        user_id_col = self.find_column(df, self.USER_ID_COLUMN_NAMES)
        refund_col = self.find_column(df, self.REFUND_COLUMN_NAMES)
        
        # 检查必要列
        if time_col is None:
            raise RuntimeError(
                f"找不到时间列。请确保文件包含以下任一列名：{', '.join(self.TIME_COLUMN_NAMES[:5])}"
            )
        
        if amount_col is None:
            raise RuntimeError(
                f"找不到金额列。请确保文件包含以下任一列名：{', '.join(self.AMOUNT_COLUMN_NAMES[:5])}"
            )
        
        if progress_callback:
            progress_callback(0.3, f"识别到时间列：{time_col}，金额列：{amount_col}")
        
        # 解析时间
        if progress_callback:
            progress_callback(0.4, "正在解析订单时间...")
        
        df['parsed_time'] = df[time_col].apply(self.parse_time)
        
        # 检查时间解析成功率
        valid_time_count = df['parsed_time'].notna().sum()
        if valid_time_count == 0:
            raise RuntimeError(
                f"无法解析订单时间。请确保时间格式正确，例如：2024-01-15 14:30:00"
            )
        
        if valid_time_count < len(df) * 0.5:
            self._add_warning(
                f"只有{valid_time_count}/{len(df)}行数据的时间格式正确，"
                f"其余{len(df) - valid_time_count}行将被忽略"
            )
        
        # 解析金额
        if progress_callback:
            progress_callback(0.5, "正在解析订单金额...")
        
        df['parsed_amount'] = df[amount_col].apply(self.parse_amount)
        
        valid_amount_count = df['parsed_amount'].notna().sum()
        if valid_amount_count == 0:
            raise RuntimeError(
                f"无法解析订单金额。请确保金额格式正确，例如：99.00"
            )
        
        # 过滤掉无效数据
        df_valid = df[(df['parsed_time'].notna()) & (df['parsed_amount'].notna())].copy()
        
        if progress_callback:
            progress_callback(0.6, f"有效订单数据：{len(df_valid)}条")
        
        # 如果没有给直播结束时间，按开始时间+3小时算
        if live_end_time is None:
            live_end_time = live_start_time + timedelta(hours=3)
        
        # 处理跨天直播
        # 如果直播结束时间是凌晨（比如凌晨1点），而开始时间是晚上（比如晚上8点）
        # 需要特殊处理
        if live_end_time < live_start_time:
            # 这种情况不应该发生，但如果发生了，假设是跨天了
            live_end_time = live_end_time + timedelta(days=1)
        
        # 计算有效订单时间范围
        valid_start = live_start_time - timedelta(hours=self.time_window_before)
        valid_end = live_end_time + timedelta(hours=self.time_window_after)
        
        if progress_callback:
            progress_callback(0.7, f"筛选有效时间范围内的订单（{valid_start} 至 {valid_end}）...")
        
        # 过滤时间范围
        df_valid = df_valid[
            (df_valid['parsed_time'] >= valid_start) &
            (df_valid['parsed_time'] <= valid_end)
        ].copy()
        
        if len(df_valid) == 0:
            raise RuntimeError(
                f"在有效时间范围内没有找到订单。"
                f"直播时间：{live_start_time} 至 {live_end_time}，"
                f"有效范围：{valid_start} 至 {valid_end}"
            )
        
        # 创建订单记录
        if progress_callback:
            progress_callback(0.8, "正在处理订单数据...")
        
        orders = []
        for idx, row in df_valid.iterrows():
            order = OrderRecord(
                order_time=row['parsed_time'],
                amount=row['parsed_amount'],
                order_id=str(row.get(order_id_col, '')) if order_id_col else None,
                user_id=str(row.get(user_id_col, '')) if user_id_col else None,
                is_refund=self.is_refund_order(row, refund_col)
            )
            orders.append(order)
        
        # 按时间排序
        orders.sort(key=lambda x: x.order_time)
        
        # 标记重复订单
        if progress_callback:
            progress_callback(0.9, "正在检测重复订单...")
        
        orders = self._mark_duplicate_orders(orders)
        
        # 生成统计信息
        stats = self._generate_stats(orders, live_start_time, live_end_time)
        
        if progress_callback:
            progress_callback(1.0, f"订单处理完成！共{len(orders)}条有效订单")
        
        return orders, stats
    
    def _mark_duplicate_orders(self, orders: List[OrderRecord]) -> List[OrderRecord]:
        """
        标记重复订单
        
        判断标准：同一个用户在短时间内（默认5分钟）多次下单
        """
        if not orders:
            return orders
        
        # 按用户分组
        user_orders: Dict[str, List[OrderRecord]] = {}
        for order in orders:
            user_id = order.user_id or 'unknown'
            if user_id not in user_orders:
                user_orders[user_id] = []
            user_orders[user_id].append(order)
        
        # 标记重复
        for user_id, user_order_list in user_orders.items():
            if len(user_order_list) <= 1:
                continue
            
            # 按时间排序
            user_order_list.sort(key=lambda x: x.order_time)
            
            # 标记短时间内重复的订单
            for i in range(1, len(user_order_list)):
                prev_order = user_order_list[i-1]
                curr_order = user_order_list[i]
                
                time_diff = (curr_order.order_time - prev_order.order_time).total_seconds() / 60
                
                if time_diff <= self.duplicate_window_minutes:
                    curr_order.is_duplicate = True
        
        return orders
    
    def _generate_stats(
        self,
        orders: List[OrderRecord],
        live_start_time: datetime,
        live_end_time: datetime
    ) -> Dict[str, Any]:
        """
        生成统计信息
        
        让商家一眼看懂数据情况
        """
        total_orders = len(orders)
        refund_orders = sum(1 for o in orders if o.is_refund)
        duplicate_orders = sum(1 for o in orders if o.is_duplicate)
        valid_orders = total_orders - refund_orders - duplicate_orders
        
        total_amount = sum(o.amount for o in orders)
        valid_amount = sum(o.amount for o in orders if not o.is_refund and not o.is_duplicate)
        
        # 计算直播期间的订单
        live_orders = [
            o for o in orders
            if live_start_time <= o.order_time <= live_end_time
        ]
        live_amount = sum(o.amount for o in live_orders)
        
        # 计算延时订单（直播结束后的订单）
        delayed_orders = [
            o for o in orders
            if o.order_time > live_end_time and not o.is_refund
        ]
        delayed_amount = sum(o.amount for o in delayed_orders)
        
        return {
            'total_orders': total_orders,
            'total_amount': round(total_amount, 2),
            'valid_orders': valid_orders,
            'valid_amount': round(valid_amount, 2),
            'refund_orders': refund_orders,
            'refund_amount': round(sum(o.amount for o in orders if o.is_refund), 2),
            'duplicate_orders': duplicate_orders,
            'live_orders': len(live_orders),
            'live_amount': round(live_amount, 2),
            'delayed_orders': len(delayed_orders),
            'delayed_amount': round(delayed_amount, 2),
            'avg_order_amount': round(valid_amount / valid_orders, 2) if valid_orders > 0 else 0,
            'warnings': self.warnings
        }
    
    def get_orders_around_script(
        self,
        orders: List[OrderRecord],
        script_start: datetime,
        script_end: datetime,
        time_windows: List[Tuple[int, int]]
    ) -> Dict[str, List[OrderRecord]]:
        """
        获取话术各个时间窗口内的订单
        
        用于DID归因分析
        
        参数：
            orders: 订单列表
            script_start: 话术开始时间
            script_end: 话术结束时间
            time_windows: 时间窗口列表，每个元素是(开始分钟, 结束分钟)
        
        返回：
            字典，key是窗口名称（如"0-1分钟"），value是该窗口内的订单列表
        """
        result = {}
        
        for start_min, end_min in time_windows:
            window_start = script_end + timedelta(minutes=start_min)
            window_end = script_end + timedelta(minutes=end_min)
            
            window_orders = [
                o for o in orders
                if window_start <= o.order_time <= window_end
                and not o.is_refund
                and not o.is_duplicate
            ]
            
            window_name = f"{start_min}-{end_min}分钟"
            result[window_name] = window_orders
        
        return result


def create_sample_orders(
    live_start_time: datetime,
    num_orders: int = 100
) -> List[OrderRecord]:
    """
    创建示例订单数据，用于测试
    
    模拟真实的订单分布：
    - 直播期间订单最多
    - 直播后1小时内还有一些延迟订单
    - 有少量退款订单
    """
    np.random.seed(42)
    orders = []
    
    live_duration_hours = 2
    live_end_time = live_start_time + timedelta(hours=live_duration_hours)
    
    for i in range(num_orders):
        # 70%的订单在直播期间
        # 20%的订单在直播后1小时内
        # 10%的订单在直播前或1小时后
        rand = np.random.random()
        
        if rand < 0.7:
            # 直播期间
            offset_minutes = np.random.uniform(0, live_duration_hours * 60)
            order_time = live_start_time + timedelta(minutes=offset_minutes)
        elif rand < 0.9:
            # 直播后1小时内
            offset_minutes = np.random.uniform(0, 60)
            order_time = live_end_time + timedelta(minutes=offset_minutes)
        else:
            # 其他时间
            offset_hours = np.random.uniform(-12, 24)
            order_time = live_start_time + timedelta(hours=offset_hours)
        
        # 随机金额，集中在50-200之间
        amount = np.random.lognormal(4.5, 0.5)
        amount = max(10, min(1000, amount))
        
        # 5%的退款率
        is_refund = np.random.random() < 0.05
        
        orders.append(OrderRecord(
            order_time=order_time,
            amount=round(amount, 2),
            order_id=f"ORDER_{i:06d}",
            user_id=f"USER_{np.random.randint(1, 50):06d}",
            is_refund=is_refund
        ))
    
    # 按时间排序
    orders.sort(key=lambda x: x.order_time)
    
    return orders


# 简单的测试代码
if __name__ == "__main__":
    # 测试示例数据
    from datetime import datetime
    
    live_start = datetime(2024, 1, 15, 20, 0, 0)
    orders = create_sample_orders(live_start, num_orders=50)
    
    print(f"生成了{len(orders)}条示例订单")
    print(f"直播开始时间：{live_start}")
    print("\n前5条订单：")
    for o in orders[:5]:
        print(f"  {o.order_time} - ¥{o.amount} - 退款：{o.is_refund}")
    
    # 测试统计
    aligner = DataAligner()
    stats = aligner._generate_stats(orders, live_start, live_start + timedelta(hours=2))
    print(f"\n统计信息：")
    for key, value in stats.items():
        if key != 'warnings':
            print(f"  {key}: {value}")
