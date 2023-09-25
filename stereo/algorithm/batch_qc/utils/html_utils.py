#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:16
# @Author  : zhangchao
# @File    : html_utils.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import base64

from lxml import etree


def embed_tabel(dataframe, tree, pos, name, is_round=True):
    """Embed Table into HTML file.

    Parameters:
    -------------------------------
    dataframe: `DataFrame`
        The data to be inserted into HTML file.
    tree:
        Element object
    pos: `str`
        The type of label to insert into the table position.
    name: `str`
        The name of label.
    is_round: bool
    """
    result = None
    col_tag = None
    columns = dataframe.columns.tolist()
    columns.insert(0, "")
    if "describe_note" in columns:
        columns.remove("describe_note")
        result = dataframe.pop("describe_note").values[0]
        col_tag = "describe_note"
    elif "summary" in columns:
        columns.remove("summary")
        result = dataframe.pop("summary").values[0]
        col_tag = "summary"
    header = tree.xpath(f"//{pos}[@class='{name}']")[0]

    table = etree.Element("table", style="align: center;"
                                         "border: 2px solid #ddd;"
                                         "text-align: center;"
                                         "border-collapse: collapse;"
                                         "width: 75%;"
                                         "margin: auto")
    # table header
    tr = etree.SubElement(table, "tr")
    for col in columns:
        th = etree.SubElement(tr, "th", style="align: center;"
                                              "border: 2px solid #ddd;"
                                              "text-align: center;"
                                              "padding: 10px;"
                                              "background-color: #507171")
        th.text = f"{col}".title()

    for idx in dataframe.index:
        tr = etree.SubElement(table, "tr")
        td = etree.SubElement(tr, "td", style="align: center;"
                                              "border: 2px solid #ddd;"
                                              "text-align: center;"
                                              "padding: 15px;")
        td.text = f"{idx}".title()
        for val in dataframe.loc[idx].values:
            td = etree.SubElement(tr, "td", style="align: center;"
                                                  "border: 2px solid #ddd;")
            if not is_round:
                td.text = f"{val}"
            else:
                try:
                    td.text = f"{val:.4f}"
                except:  # noqa
                    td.text = f"{val}"

    if result is not None:
        tr = etree.SubElement(table, "tr")
        td = etree.SubElement(tr, "td", style="align: center;"
                                              "border: 2px solid #ddd;"
                                              "text-align: center;"
                                              "padding: 15px;"
                                              "background-color: #F0F0F0")
        if col_tag == "describe_note":
            td.text = "describe note".title()
        elif col_tag == "summary":
            td.text = "summary".title()
        td = etree.SubElement(tr, "td", style="align: center;"
                                              "border: 2px solid #ddd;"
                                              "text-align: center;"
                                              "padding: 15px;"
                                              "white-space:pre-line;"
                                              "word-wrap:break-word;"
                                              "word-break: break-all;"
                                              "background-color: #F0F0F0",
                              colspan=f"{len(columns) - 1}")
        td.text = f"{result}"
    header.addnext(table)


def embed_table_imgs(buffer_dict, tree, pos, class_name):
    header = tree.xpath(f"//{pos}[@class='{class_name}']")[0]

    table = etree.Element("table", style="align: center;"
                                         "border: 0px;"
                                         "text-align: center;"
                                         "border-collapse: collapse;"
                                         "table-layout:automatic;"
                                         "width: 100%;"
                                         "margin: auto")
    dict_keys = list(buffer_dict.keys())
    for i in range(0, len(dict_keys), 3):
        n_row = (len(dict_keys) - i) if (len(dict_keys) - i) < 3 else 3
        tr = etree.SubElement(table, "tr")
        for j in range(n_row):
            td = etree.SubElement(tr, "td", style="align: center;"
                                                  "text-align: center;"
                                                  "padding: 15px;")
            td.text = f"{dict_keys[i + j]}".title()
        tr = etree.SubElement(table, "tr")
        for j in range(n_row):
            td = etree.SubElement(tr, "td", style="align: center;"
                                                  "text-align: center;"
                                                  "padding: 15px;")
            byte_data = buffer_dict[dict_keys[i + j]].getvalue()
            fig_src = f"data:image/png;base64,{base64.b64encode(byte_data).decode('UTF-8')}"
            img = etree.SubElement(td, "img", src=fig_src, width="568", height="320")  # noqa
    header.addnext(table)


def embed_text(tree, pos, name, text):
    header = tree.xpath(f"//{pos}[@class='{name}']")[0]
    header.text = text
