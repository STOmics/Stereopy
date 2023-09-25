# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 09:09
# @Author  : liuxiaobin
# @File    : database_utils.py
# @Version：V 0.1
# @desc :


import pandas as pd
from sqlalchemy import (
    Table,
    MetaData,
    ForeignKeyConstraint,
)
from sqlalchemy.engine import reflection
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.ddl import DropConstraint
from sqlalchemy.sql.ddl import DropTable

from stereo.log_manager import logger


class Database:
    def __init__(self, engine):
        self.engine = engine
        self.established_session = None
        self.base = declarative_base()
        self.base_model = None

    @property
    def session(self):
        if not self.established_session:
            session = sessionmaker(bind=self.engine)
            self.established_session = session()

        return self.established_session

    def create_all(self):
        self.base_model.metadata.create_all(self.engine)

    def drop_everything(self):
        conn = self.engine.connect()

        trans = conn.begin()

        inspector = reflection.Inspector.from_engine(self.engine)

        metadata = MetaData()

        tbs = []
        all_fks = []

        for table_name in inspector.get_table_names():
            fks = []
            for fk in inspector.get_foreign_keys(table_name):
                if not fk['name']:
                    continue
                fks.append(
                    ForeignKeyConstraint((), (), name=fk['name'])
                )
            t = Table(table_name, metadata, *fks)
            tbs.append(t)
            all_fks.extend(fks)

        for fkc in all_fks:
            conn.execute(DropConstraint(fkc))

        for table in tbs:
            conn.execute(DropTable(table))

        trans.commit()


class DatabaseManager:
    def __init__(self, repositories, db=None):
        self._repositories = repositories
        self.database = db

    def add_repository(self, repository):
        if not self._repositories:
            self._repositories = {}

        self._repositories[repository.name] = repository

    def get_repository(self, repository_name):
        return self._repositories[repository_name](self)

    def get_column_table_names(self, model_name: str):
        def get_model():
            for c in self.database.base_model._decl_class_registry.values():
                if hasattr(c, '__tablename__') and c.__tablename__ == model_name:
                    return c

        colum_names = self.database.session.query(get_model()).statement.columns
        colum_names = [p.name for p in colum_names]
        return colum_names


class Repository:
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager

    @staticmethod
    def _blend_column(original_df: pd.DataFrame, multidata_df: pd.DataFrame, original_column_name: list,
                      db_column_name: list, table_name: str, number: int) -> pd.DataFrame:
        """

        :param original_df:
        :param multidata_df:
        :param original_column_name:
        :param db_column_name:
        :param table_name:
        :param number:
        :return:
        """
        interaction_df = pd.merge(original_df, multidata_df, left_on=original_column_name, right_on=db_column_name,
                                  indicator=True, how='outer')
        interaction_df.rename(index=str, columns={'id_%s' % table_name: '%s_%s_id' % (table_name, number)},
                              inplace=True)

        interaction_df = interaction_df[
            (interaction_df['_merge'] == 'both') | (interaction_df['_merge'] == 'left_only')]  # 这不就等于左merge？
        interaction_df.rename(
            index=str,
            columns={'_merge': '_merge_%s' % number, db_column_name: db_column_name + '_%s' % number},
            inplace=True
        )

        return interaction_df

    @staticmethod
    def blend_dataframes(left_df, left_column_names, right_df, db_column_name, db_table_name, quiet=False):
        """

        :param left_df:
        :param left_column_names:
        :param right_df:
        :param db_column_name:
        :param db_table_name:
        :param quiet:
        :return:
        """
        result_df = left_df.copy()

        if not quiet and db_column_name in left_df.columns:
            logger.debug('WARNING | BLENDING: column "%s" already exists in orginal df' % db_column_name)

        unique_slug = '_EDITNAME'
        unique_original_column_names = [("%s%s" % (column_name, unique_slug)) for column_name in left_column_names]

        result_df.rename(index=str, columns=dict(zip(left_column_names, unique_original_column_names)),
                         inplace=True)

        not_existent_proteins = []

        for i in range(0, len(unique_original_column_names)):
            result_df = Repository._blend_column(result_df, right_df, unique_original_column_names[i],
                                                 db_column_name,
                                                 db_table_name, i + 1)

            not_existent_proteins = not_existent_proteins + result_df[result_df['_merge_%s' % (i + 1)] == 'left_only'][
                unique_original_column_names[i]].drop_duplicates().tolist()
        not_existent_proteins = list(set(not_existent_proteins))

        for i in range(1, len(unique_original_column_names) + 1):
            result_df = result_df[(result_df['_merge_%s' % i] == 'both')]

        result_df.drop(['_merge_%s' % merge_column for merge_column in
                        range(1, len(unique_original_column_names) + 1)] + unique_original_column_names, axis=1,
                       inplace=True)

        if len(left_column_names) == 1:
            result_df.rename(index=str, columns={'%s_1' % db_column_name: db_column_name,
                                                 '%s_1_id' % db_table_name: '%s_id' % db_table_name}, inplace=True)

        if not quiet and not_existent_proteins:
            logger.warning('WARNING | BLENDING: THIS %s DIDNT EXIST IN %s' % (db_column_name, db_table_name))
            logger.warning(not_existent_proteins)

        return result_df


def remove_not_defined_columns(data_frame: pd.DataFrame, defined_columns: list) -> pd.DataFrame:
    """
    Remove undefined columns from dataframe

    This method receives a dataframe and a list of columns and drops all
    the columns that are not in the given list.

    Parameters
    ----------
    data_frame: pd.DataFrame
        Original DataFrame.
    defined_columns: list.
        List of columns to keep.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the columns specified in defined_columns.
    """
    data_frame_keys = list(data_frame.keys())

    for key in data_frame_keys:
        if key not in defined_columns:
            data_frame.drop(key, axis=1, inplace=True)

    return data_frame
