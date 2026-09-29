import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import epub_chapters, reader_assets


class EpubTocTests(unittest.TestCase):
    def build(self, navigation, *, nav=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / 'book.epub'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('META-INF/container.xml', '<container><rootfiles><rootfile full-path="O/content.opf"/></rootfiles></container>')
            item = ('<item id="toc" href="nav.xhtml" properties="nav" media-type="application/xhtml+xml"/>' if nav else
                    '<item id="toc" href="toc.ncx" media-type="application/x-dtbncx+xml"/>')
            archive.writestr('O/content.opf', '<package><manifest>' + item + '<item id="a" href="book.xhtml" media-type="application/xhtml+xml"/><item id="b" href="split.xhtml" media-type="application/xhtml+xml"/></manifest><spine toc="toc"><itemref idref="a"/><itemref idref="b"/></spine></package>')
            archive.writestr('O/nav.xhtml' if nav else 'O/toc.ncx', navigation)
            archive.writestr('O/book.xhtml', '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Unknown Text</title></head><body><h1 id="卷一">第一卷</h1><p>前文</p><h2 id="day">一月三日</h2><p>正文</p></body></html>')
            archive.writestr('O/split.xhtml', '<html><body><p>没有独立目录项的续页正文</p></body></html>')
        return epub_chapters.build_bundle(source, root / 'bundle')

    def test_ncx_keeps_nested_targets_in_same_file_and_recovers_unknown_label(self):
        result = self.build('<ncx><navMap><navPoint><navLabel><text>第一卷</text></navLabel><content src="book.xhtml#%E5%8D%B7%E4%B8%80"/><navPoint><navLabel><text>Unknown Text</text></navLabel><content src="book.xhtml#day"/></navPoint></navPoint></navMap></ncx>')
        self.assertEqual(result['toc'], [
            dict(title='第一卷', chapter=1, fragment='卷一', depth=0),
            dict(title='一月三日', chapter=1, fragment='day', depth=1)])
        self.assertEqual(len(result['chapters']), 2)

    def test_epub3_uses_toc_only_and_keeps_unlinked_group(self):
        result = self.build('<html xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="landmarks"><ol><li><a href="split.xhtml">正文</a></li></ol></nav><nav epub:type="toc"><ol><li><span>日记</span><ol><li><a href="book.xhtml#day">一月三日</a></li></ol></li></ol></nav></body></html>', nav=True)
        self.assertEqual(result['toc'], [dict(title='日记', chapter=1, fragment='day', depth=0),
                                         dict(title='一月三日', chapter=1, fragment='day', depth=1)])

    def test_unrecoverable_navigation_fails_instead_of_publishing_placeholder(self):
        with self.assertRaisesRegex(ValueError, 'cannot be recovered'):
            self.build('<ncx><navMap><navPoint><navLabel><text>Unknown Text</text></navLabel><content src="book.xhtml#missing"/></navPoint></navMap></ncx>')

    def test_missing_navigation_fragment_falls_back_to_chapter_start(self):
        result = self.build('<ncx><navMap><navPoint><navLabel><text>第一卷</text></navLabel><content src="book.xhtml#missing"/></navPoint></navMap></ncx>')
        self.assertEqual(result['toc'], [dict(title='第一卷', chapter=1, fragment='', depth=0)])

    def test_manifest_rejects_bad_navigation_target(self):
        result = self.build('<ncx><navMap/></ncx>')
        result['toc'] = [dict(title='错误', chapter=3, depth=0, fragment='')]
        with self.assertRaisesRegex(ValueError, 'TOC entry'):
            reader_assets.validate_chapter_manifest(result)


if __name__ == '__main__':
    unittest.main()
