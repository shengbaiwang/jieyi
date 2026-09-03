import unittest

from jieyi.ingestion.epub_navigation import parse_xml_resource


class EpubXmlTests(unittest.TestCase):
    def test_plain_doctype_accepts_xml_whitespace_and_encodings(self):
        for doctype in ("<!DOCTYPE html>", "<!DOCTYPE\nhtml\t>"):
            for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
                with self.subTest(doctype=doctype, encoding=encoding):
                    root = parse_xml_resource(
                        (doctype + "<html><body>Arrière-pays</body></html>").encode(encoding),
                        "chapter.xhtml",
                    )
                    self.assertEqual(root.find("body").text, "Arrière-pays")

    def test_declaration_text_in_comments_and_cdata_is_not_executed(self):
        data = (
            b'<!-- <!DOCTYPE html SYSTEM "file:///unused.dtd"> -->'
            b'<!DOCTYPE html><html><body><![CDATA[<!ENTITY example "text">]]></body></html>'
        )
        root = parse_xml_resource(data, "chapter.xhtml")
        self.assertEqual(root.find("body").text, '<!ENTITY example "text">')

    def test_rejects_subsets_entities_and_untrusted_external_identifiers(self):
        declarations = (
            '<!DOCTYPE html SYSTEM "file:///unused.dtd">',
            '<!DOCTYPE html SYSTEM "https://example.invalid/external.dtd">',
            '<!DOCTYPE html PUBLIC "untrusted" "https://example.invalid/external.dtd">',
            '<!DOCTYPE html []>',
            '<!DOCTYPE html [<!ENTITY local "expanded">]>',
            '<!DOCTYPE html [<!ENTITY xxe SYSTEM "file:///unused.txt">]>',
            (
                '<!DOCTYPE html [<!ENTITY % remote SYSTEM "https://example.invalid/e.dtd">'
                '%remote;]>'
            ),
            '<!DOCTYPE html [<!ENTITY a "ha"><!ENTITY b "&a;&a;">]>',
            '<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" "file:///unused.dtd">',
        )
        for declaration in declarations:
            for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
                with (
                    self.subTest(declaration=declaration, encoding=encoding),
                    self.assertRaisesRegex(ValueError, "DTD or entity.*chapter.xhtml"),
                ):
                    parse_xml_resource(
                        (declaration + "<html><body>Text</body></html>").encode(encoding),
                        "chapter.xhtml",
                    )

    def test_rejects_malformed_and_duplicate_declarations(self):
        for data in (
            b"<!DOCTYPE html",
            b"<!DOCTYPE html><!DOCTYPE html><html/>",
            b"<html><!DOCTYPE html></html>",
            b"<!ENTITY xxe 'text'><html/>",
        ):
            with (
                self.subTest(data=data),
                self.assertRaisesRegex(ValueError, "Invalid XML in chapter.xhtml"),
            ):
                parse_xml_resource(data, "chapter.xhtml")

    def test_named_html_entities_preserve_text_and_xml_escaping(self):
        root = parse_xml_resource(
            b"<!DOCTYPE html><html><p>&nbsp;&eacute;&amp;&lt;&nvlt;&#233;</p></html>",
            "chapter.xhtml",
        )
        self.assertEqual(root.find("p").text, "\u00a0é&<<\u20d2é")

    def test_unknown_entities_are_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid XML in chapter.xhtml"):
            parse_xml_resource(b"<!DOCTYPE html><html>&unknown;</html>", "chapter.xhtml")
