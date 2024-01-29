# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 08:49
# @Author  : liuxiaobin
# @File    : sqlalchemy_model.py
# @Version：V 0.1
# @desc : 建立数据库里六张表与python class的映射model


from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Complex(Base):
    __tablename__ = 'complex_table'
    # primary key
    id_complex = Column(Integer, nullable=False, primary_key=True)
    # columns
    pdb_structure = Column(String)
    pdb_id = Column(String)
    stoichiometry = Column(String)
    comments_complex = Column(String)
    # foreign key
    complex_multidata_id = Column(Integer, ForeignKey('multidata_table.id_multidata'), nullable=False, unique=True)


class ComplexComposition(Base):
    __tablename__ = 'complex_composition_table'
    # primary key
    id_complex_composition = Column(Integer, nullable=False, primary_key=True)
    # columns
    total_protein = Column(Integer)
    # foreign key
    complex_multidata_id = Column(Integer, ForeignKey('multidata_table.id_multidata'), nullable=False)
    protein_multidata_id = Column(Integer, ForeignKey('multidata_table.id_multidata'), nullable=False)


class Gene(Base):
    __tablename__ = 'gene_table'
    # primary key
    id_gene = Column(Integer, nullable=False, primary_key=True)
    # columns
    ensembl = Column(String, nullable=False)
    gene_name = Column(String, nullable=False)
    hgnc_symbol = Column(String)
    # foreign key
    protein_id = Column(Integer, ForeignKey('protein_table.id_protein'), nullable=False)


class Protein(Base):
    __tablename__ = 'protein_table'
    # primary key
    id_protein = Column(Integer, nullable=False, primary_key=True)
    # columns
    protein_name = Column(String)
    tags = Column(String)
    tags_reason = Column(String)
    tags_description = Column(String)
    # pfam = Column(String)  # pfam no longer exists in the database
    # foreign key
    protein_multidata_id = Column(Integer, ForeignKey('multidata_table.id_multidata'), unique=True, nullable=False)
    # relation
    gene = relationship('Gene', backref='gene_table', lazy='subquery')


class Multidata(Base):
    __tablename__ = 'multidata_table'
    # primary key
    id_multidata = Column(Integer, nullable=False, primary_key=True)
    # columns
    name = Column(String, nullable=False, unique=True)
    receptor = Column(Boolean)
    receptor_desc = Column(String)
    other = Column(Boolean)
    other_desc = Column(String)
    secreted_highlight = Column(Boolean)
    secreted_desc = Column(String)
    transmembrane = Column(Boolean)
    secreted = Column(Boolean)
    peripheral = Column(Boolean)
    integrin = Column(Boolean)
    is_complex = Column(Boolean)
    # relation
    protein = relationship('Protein', backref='protein_table', lazy='subquery')
    complex = relationship('Complex', backref='complex_table', lazy='subquery')


class Interaction(Base):
    __tablename__ = 'interaction_table'
    # primary key
    id_interaction = Column(Integer, nullable=False, primary_key=True)
    id_cp_interaction = Column(String, nullable=False, unique=True)
    # columns
    source = Column(String)
    annotation_strategy = Column(String)
    # foreign key
    multidata_1_id = Column(Integer, ForeignKey('multidata_table.id_multidata'), nullable=False)
    multidata_2_id = Column(Integer, ForeignKey('multidata_table.id_multidata'), nullable=False)
