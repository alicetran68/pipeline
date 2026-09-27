# 林一章版 PDF 字体修复

`reader_assets.KNOWN_GBK_PDFS` 精确列出 Teachers 的 11 个分册和合订本；
`GBK_PDF_REPAIR_FOLDERS` 仅扩展到抽查确认存在同类无嵌入 GBK 字体的
马恩新版/选集、列宁选集、斯大林旧版全集/选集、毛选和鲁迅系列。
马恩旧版、列宁新版等已有正常嵌入字体/文字的系列不进入修复队列。
这些文件的正文包含 GBK 双字节字符，字体却声明为单字节 WinAnsi TrueType。
分册附带的单字节 ToUnicode 表与正文不匹配；合订本还丢失了 FontDescriptor
中的中文字体名及 ToUnicode；其他同类文件也可能没有 ToUnicode。
当前 PDF.js 可出现韩文、方框或西文乱码。

`gbk-font-repair-v1` 生成独立的 Reader PDF，使用 Type0、GBK-EUC-H、Adobe-GB1
及显式 GBK→Unicode 映射，修复宋体、黑体、仿宋、楷体。字体依旧使用系统替代，
不将正文栅格化。原始下载地址保持可用。源文件、每页内容流、图像与页序由文档克隆
保留；字体定义替换为有效的双字节定义。不要把这个有明确来源范围的修复当作通用
PDF 编码猜测器，也不要覆盖有效的嵌入字体或有效的双字节 ToUnicode。

队列通过 `source_conversion_contract` 选择这些文件，普通 Reader Assets 的构建、
内容哈希对象路径、manifest 和 search sidecar 发布流程均可复用。发布前使用同一
source revision 构建，并通过现有 publisher 生成完整的 manifest/sidecar 世代。

单文件生成：

```bash
python3 -B scripts/repair_gbk_pdf.py input.pdf repaired.pdf
```

快速回归：

```bash
python3 -B -m unittest tests.test_repair_gbk_pdf tests.test_lin_pdf_text tests.test_reader_assets -v
```

测试覆盖来源范围、字体判定、嵌入字体及有效字符映射的保护、原件不覆盖、共享字体、
页面内容与几何、标题及目录目标保留；安装 Poppler 时还会独立验证中文和 ASCII 提取。

真实文件验收应包括：全部页面内容流及页面尺寸对比；每册目录页、中间页和末页在
实际 PDF.js 中渲染；中文文本与原始 GBK 字节独立解码结果对比；发布后旧阅读 ID
指向修复版、Range 响应正确及原件下载地址仍可用。合订本共有 7,302 页，不能仅验证
封面图片或一两个分册。这里的页数为 2026-09-14 检查的源版本
`1d03396c8365f9ad57e9dc316a1838991c862ac0`，不代表未来的源版本。
