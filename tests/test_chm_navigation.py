import tempfile
import unittest
import zipfile
from lxml import etree
from ebooklib import epub
from pathlib import Path
from scripts.chm_navigation import SitemapParser, SourcePaths, flatten, navigation, repair_epub


class ChmNavigationTests(unittest.TestCase):
    def test_chm_keeps_native_tree_instead_of_flat_chapter_sidecar(self):
        from scripts.reader_assets import needs_epub_chapters
        self.assertTrue(needs_epub_chapters('chm','epub',1))

    def test_repaired_encoded_anchor_and_linked_full_size_image(self):
        from scripts.chm_resources import repair_images, repair_links
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'page.htm').write_text('<a name="中文">Text</a><a href="#中文">Jump</a><a href="PIC.PNG">Image</a>')
            Image.new('RGB',(4,4),'red').save(source/'pic.png')
            book=epub.EpubBook();book.set_identifier('test');book.set_title('Test');book.set_language('en')
            chapter=epub.EpubHtml(title='Page',file_name='page.xhtml',content='<p id="%E4%B8%AD%E6%96%87">Text</p><a href="#%E4%B8%AD%E6%96%87">Jump</a><a href="PIC.PNG">Image</a>')
            book.add_item(chapter);book.spine=[chapter];book.toc=[chapter];book.add_item(epub.EpubNcx());book.add_item(epub.EpubNav())
            path=root/'book.epub';epub.write_epub(str(path),book)
            mapping={'page.htm':['EPUB/page.xhtml']}
            self.assertEqual(repair_images(path,source,mapping)['added'],1)
            self.assertFalse(repair_links(path,source,mapping)['missing'])
            with zipfile.ZipFile(path) as z:
                doc=etree.fromstring(z.read('EPUB/page.xhtml'))
                links=doc.xpath('//*[local-name()="a"]/@href')
                self.assertEqual(links[0],'page.xhtml#%25E4%25B8%25AD%25E6%2596%2587')
                self.assertTrue(links[1].startswith('chm-images/'))

    def test_static_menu_restores_groups_without_evaluating_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'js').mkdir();(root/'txt').mkdir()
            (root/'js/page.js').write_text("var pages=[]; pages[0]=['a','(1)','2','First']; pages[1]=['b','(2)','2']; throw Error('must not execute');")
            for name in ['a','b']:(root/'txt'/ (name+'.txt')).write_text('body')
            result,_=navigation(root)
            self.assertEqual([(n['label'],d) for n,d in flatten(result['nodes'])],[('First',0),('(1)',1),('(2)',1)])
            self.assertEqual(result['nodes'][0]['children'][1]['source'],'txt/b.txt')

    def test_script_chapter_preserves_declared_heading_and_body(self):
        from scripts.chm_static_navigation import script_page
        title,body=script_page('var makechm_a_title="Heading";var makechm_a_content="<p>Body</p>";'
            'document.write("<h1>"+makechm_a_title+"</h1>");document.write(makechm_a_content);')
        self.assertEqual((title,body),('Heading','<h1>Heading</h1><p>Body</p>'))
        with self.assertRaises(ValueError):
            script_page('var makechm_a_content="Body";document.write(fetch("https://example.org"));')

    def test_html_contents_preserves_all_targets_beyond_fifty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'index.html').write_text('<center>上卷 正文</center>'+''.join(f'<a href="{i}.htm">标题{i}</a>' for i in range(65)))
            for i in range(65):(root/f'{i}.htm').write_text('body')
            result,_=navigation(root)
            self.assertEqual(len(result['nodes'][0]['children']),65)
            self.assertEqual(result['nodes'][0]['children'][-1]['label'],'标题64')

    def test_groups_hierarchy_apostrophes_and_repeated_anchor_targets(self):
        parser = SitemapParser()
        parser.feed('''<object type="text/site properties"><param name="Name" value="ignored"></object>
          <ul><li><object><param name="Name" value="Group's title"></object><ul>
          <li><object><param name="Name" value="One &amp; two"><param name="Local" value="a.htm#one"></object>
          <li><object><param name="Name" value="Three"><param name="Local" value="a.htm#two"></object>
          </ul><li><object><param name="Name" value="End"><param name="Local" value="b.htm"></object></ul>''')
        self.assertEqual([(n['label'],d) for n,d in flatten(parser.nodes)],
                         [("Group's title",0),('One & two',1),('Three',1),('End',0)])
        self.assertEqual([n['href'] for n,d in flatten(parser.nodes)], ['', 'a.htm#one', 'a.htm#two','b.htm'])

    def test_windows_paths_percent_encoding_and_ambiguity(self):
        paths=SourcePaths(['a/b#c.htm','images/X.png','A.htm','a.HTM'])
        self.assertEqual(paths.resolve('a/toc.hhc','b%23c.htm#section'),('a/b#c.htm','section','exact'))
        self.assertEqual(paths.resolve('a/page.htm','..\\Images\\x.png'),('images/X.png','','casefold'))
        self.assertEqual(paths.resolve('','a.htm')[2],'ambiguous')
        self.assertEqual(paths.resolve('','../../escape')[2],'outside')

    def test_multiple_distinct_hhc_trees_are_not_silently_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['one','two']:
                (root/(name+'.hhc')).write_text(f'<object><param name="Name" value="{name}"></object>')
            with self.assertRaisesRegex(ValueError,'multiple source HHC'):
                navigation(root)

    def test_repair_keeps_chapter_bytes_and_restores_source_tree_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source';source.mkdir()
            (source/'toc.hhc').write_text('<ul><li><object><param name="Name" value="Volume"></object><ul>'
                '<li><object><param name="Name" value="Original Two"><param name="Local" value="two.htm#part"></object>'
                '<li><object><param name="Name" value="Original One"><param name="Local" value="one.htm"></object></ul></ul>')
            for name in ['one','two']:(source/(name+'.htm')).write_text('<p id="part">'+name+'</p>')
            book=epub.EpubBook();book.set_identifier('test');book.set_title('Test');book.set_language('en')
            chapters=[]
            for name in ['one','two']:
                c=epub.EpubHtml(title='Wrong '+name,file_name=name+'.xhtml',content='<p id="part">'+name+'</p>')
                book.add_item(c);chapters.append(c)
            book.toc=chapters;book.spine=chapters;book.add_item(epub.EpubNcx());book.add_item(epub.EpubNav())
            path=root/'book.epub';epub.write_epub(str(path),book)
            with zipfile.ZipFile(path) as archive:
                original={n:archive.read(n) for n in ['EPUB/one.xhtml','EPUB/two.xhtml']}
            report=repair_epub(path,source,{'one.htm':['EPUB/one.xhtml'],'two.htm':['EPUB/two.xhtml']})
            self.assertFalse(report['spine_preserved'])
            with zipfile.ZipFile(path) as archive:
                for name,data in original.items():self.assertEqual(archive.read(name),data)
                nav=etree.fromstring(archive.read('EPUB/nav.xhtml'))
                self.assertEqual(nav.xpath('//*[local-name()="span"]/text()'),['Volume'])
                self.assertEqual(nav.xpath('//*[local-name()="a"]/@href'),['two.xhtml#part','one.xhtml'])
            self.assertEqual(report['spine'],['EPUB/two.xhtml','EPUB/one.xhtml'])
