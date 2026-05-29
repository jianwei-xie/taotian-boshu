"""
报告生成器

生成PDF格式的分析报告，商家可以打印出来给主播看
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# 尝试导入reportlab，如果没有则使用简单的文本报告
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True

    # 注册中文字体（Windows系统自带）
    _CN_FONTS_REGISTERED = False
    try:
        _font_paths = [
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
        ]
        for _fp in _font_paths:
            if os.path.exists(_fp):
                pdfmetrics.registerFont(TTFont('CNFont', _fp))
                _CN_FONTS_REGISTERED = True
                break
    except Exception:
        pass
except ImportError:
    HAS_REPORTLAB = False


class ReportGenerator:
    """
    报告生成器
    
    生成PDF格式的分析报告
    """
    
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "data" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_pdf_report(self, data: Dict[str, Any]) -> str:
        """
        生成PDF报告
        
        参数：
            data: 包含所有分析结果的字典
            
        返回：
            PDF文件路径
        """
        if HAS_REPORTLAB:
            return self._generate_pdf_with_reportlab(data)
        else:
            return self._generate_text_report(data)
    
    def _generate_pdf_with_reportlab(self, data: Dict[str, Any]) -> str:
        """使用reportlab生成PDF"""
        # 生成输出文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"淘天播术-电商直播话术军师分析报告_{timestamp}.pdf"
        
        # 创建PDF文档
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # 准备样式
        styles = getSampleStyleSheet()
        _cn = 'CNFont' if _CN_FONTS_REGISTERED else 'Helvetica'

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#FF6B6B'),
            spaceAfter=30,
            alignment=1,  # 居中
            fontName=_cn
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            spaceBefore=12,
            fontName=_cn
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            fontName=_cn
        )
        
        # 构建内容
        story = []
        
        # 标题
        story.append(Paragraph("淘天播术-电商直播话术军师 - 直播话术分析报告", title_style))
        story.append(Paragraph(f"生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}", normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        # 核心数据概览
        story.append(Paragraph("一、核心数据概览", heading_style))
        
        stats = data.get('order_stats', {})
        overview_data = [
            ['指标', '数值'],
            ['有效订单数', f"{stats.get('valid_orders', 0)}"],
            ['有效GMV', f"¥{stats.get('valid_amount', 0):,.2f}"],
            ['平均客单价', f"¥{stats.get('avg_order_amount', 0):.2f}"],
            ['直播期间订单', f"{stats.get('live_orders', 0)}"],
            ['延时订单', f"{stats.get('delayed_orders', 0)}"],
        ]
        
        overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FF6B6B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), _cn),
            ('FONTNAME', (0, 1), (-1, -1), _cn),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.3*inch))
        
        # 高转化话术TOP10
        story.append(Paragraph("二、高转化话术TOP10", heading_style))
        
        attribution_results = data.get('attribution_results', [])
        top_scripts = sorted(
            attribution_results,
            key=lambda x: x.incremental_gmv,
            reverse=True
        )[:10]
        
        scripts_data = [['排名', '标签', '话术内容', '增量GMV', '提升率']]
        
        for i, script in enumerate(top_scripts, 1):
            scripts_data.append([
                str(i),
                script.label,
                script.text[:30] + '...' if len(script.text) > 30 else script.text,
                f"¥{script.incremental_gmv:.0f}",
                f"{script.lift_rate*100:.1f}%"
            ])
        
        scripts_table = Table(scripts_data, colWidths=[0.5*inch, 0.8*inch, 2.5*inch, 0.8*inch, 0.7*inch])
        scripts_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4ECDC4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), _cn),
            ('FONTNAME', (0, 1), (-1, -1), _cn),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(scripts_table)
        story.append(Spacer(1, 0.3*inch))
        
        # 最优话术组合策略
        story.append(PageBreak())
        story.append(Paragraph("三、最优话术组合策略", heading_style))
        
        optimization = data.get('optimization_result')
        if optimization:
            story.append(Paragraph("推荐话术配比：", normal_style))
            story.append(Spacer(1, 0.1*inch))
            
            ratio_data = [['话术类型', '推荐占比']]
            for label, ratio in sorted(
                optimization.optimal_ratio.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if ratio >= 0.05:
                    ratio_data.append([label, f"{ratio*100:.1f}%"])
            
            ratio_table = Table(ratio_data, colWidths=[3*inch, 2*inch])
            ratio_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F7DC6F')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), _cn),
                ('FONTNAME', (0, 1), (-1, -1), _cn),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(ratio_table)
            story.append(Spacer(1, 0.3*inch))
            
            # 分阶段建议
            story.append(Paragraph("分阶段话术策略：", normal_style))
            story.append(Spacer(1, 0.1*inch))
            
            for stage_name, advice in optimization.stage_advice.items():
                story.append(Paragraph(f"<b>{stage_name}</b>（{advice['duration_minutes']:.0f}分钟）：{advice['description']}", normal_style))
                story.append(Spacer(1, 0.05*inch))
        
        # 关键洞察
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("四、关键洞察与建议", heading_style))
        
        if optimization and optimization.insights:
            for insight in optimization.insights:
                story.append(Paragraph(f"• {insight}", normal_style))
                story.append(Spacer(1, 0.05*inch))
        
        # 生成PDF
        doc.build(story)
        
        return str(output_path)
    
    def _generate_text_report(self, data: Dict[str, Any]) -> str:
        """
        生成文本报告（当reportlab不可用时）
        
        返回文本文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"淘天播术-电商直播话术军师分析报告_{timestamp}.txt"
        
        lines = []
        lines.append("=" * 60)
        lines.append("淘天播术-电商直播话术军师 - 直播话术分析报告")
        lines.append(f"生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        lines.append("=" * 60)
        lines.append("")
        
        # 核心数据概览
        lines.append("一、核心数据概览")
        lines.append("-" * 40)
        
        stats = data.get('order_stats', {})
        lines.append(f"有效订单数：{stats.get('valid_orders', 0)}")
        lines.append(f"有效GMV：¥{stats.get('valid_amount', 0):,.2f}")
        lines.append(f"平均客单价：¥{stats.get('avg_order_amount', 0):.2f}")
        lines.append(f"直播期间订单：{stats.get('live_orders', 0)}")
        lines.append(f"延时订单：{stats.get('delayed_orders', 0)}")
        lines.append("")
        
        # 高转化话术TOP10
        lines.append("二、高转化话术TOP10")
        lines.append("-" * 40)
        
        attribution_results = data.get('attribution_results', [])
        top_scripts = sorted(
            attribution_results,
            key=lambda x: x.incremental_gmv,
            reverse=True
        )[:10]
        
        for i, script in enumerate(top_scripts, 1):
            lines.append(f"\n#{i} [{script.label}]")
            lines.append(f"话术：{script.text}")
            lines.append(f"增量GMV：¥{script.incremental_gmv:.2f} | 提升率：{script.lift_rate*100:.1f}%")
        
        lines.append("")
        
        # 最优话术组合策略
        lines.append("三、最优话术组合策略")
        lines.append("-" * 40)
        
        optimization = data.get('optimization_result')
        if optimization:
            lines.append("\n推荐话术配比：")
            for label, ratio in sorted(
                optimization.optimal_ratio.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                if ratio >= 0.05:
                    lines.append(f"  {label}: {ratio*100:.1f}%")
            
            lines.append("\n分阶段话术策略：")
            for stage_name, advice in optimization.stage_advice.items():
                lines.append(f"\n{stage_name}（{advice['duration_minutes']:.0f}分钟）：{advice['description']}")
                for label, ratio in advice['recommended_ratio'].items():
                    duration = advice['recommended_duration'][label]
                    lines.append(f"  - {label}: {ratio*100:.0f}%（约{duration:.0f}分钟）")
        
        lines.append("")
        
        # 关键洞察
        lines.append("四、关键洞察与建议")
        lines.append("-" * 40)
        
        if optimization and optimization.insights:
            for insight in optimization.insights:
                lines.append(f"• {insight}")
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("报告生成完毕，感谢使用淘天播术-电商直播话术军师！")
        lines.append("=" * 60)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return str(output_path)


# 简单的测试代码
if __name__ == "__main__":
    # 创建示例数据
    from core import create_sample_attribution_results, ScriptOptimizer
    from core.did_attribution import DIDAttributor
    
    attributor = DIDAttributor()
    attribution_results = create_sample_attribution_results()
    label_summaries = attributor.summarize_by_label(attribution_results)
    
    optimizer = ScriptOptimizer()
    optimization_result = optimizer.optimize(label_summaries, live_duration_minutes=120)
    
    sample_data = {
        'order_stats': {
            'valid_orders': 95,
            'valid_amount': 14250.0,
            'avg_order_amount': 150.0,
            'live_orders': 70,
            'delayed_orders': 20
        },
        'attribution_results': attribution_results,
        'label_summaries': label_summaries,
        'optimization_result': optimization_result
    }
    
    generator = ReportGenerator()
    report_path = generator.generate_pdf_report(sample_data)
    
    print(f"报告已生成：{report_path}")
