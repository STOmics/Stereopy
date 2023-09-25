# -*- coding: utf-8 -*-
# @Time    : 2022/12/23 08:59
# @Author  : liuxiaobin
# @File    : sqlalchemy_repository.py
# @Version：V 0.1
# @desc :


import pandas as pd
from sqlalchemy import or_

from stereo.algorithm.cell_cell_communication.utils.database_utils import (
    Repository,
    remove_not_defined_columns
)
from stereo.algorithm.cell_cell_communication.utils.sqlalchemy_model import (
    Complex,
    ComplexComposition,
    Gene,
    Multidata,
    Protein,
    Interaction
)
from stereo.log_manager import logger


class ComplexRepository(Repository):
    name = 'complex'

    def get_all(self) -> pd.DataFrame:
        """
        259
        :return:
        """
        query = self.database_manager.database.session.query(Complex)
        result = pd.read_sql(query.statement, self.database_manager.database.engine)

        return result

    def get_all_expanded(self) -> pd.DataFrame:
        query = self.database_manager.database.session.query(Complex, Multidata).join(Multidata)
        result = pd.read_sql(query.statement, self.database_manager.database.engine)

        return result

    def get_all_compositions(self) -> pd.DataFrame:
        query = self.database_manager.database.session.query(ComplexComposition)
        result = pd.read_sql(query.statement, self.database_manager.database.engine)

        return result

    def get_all_compositions_expanded(self, include_gene: bool = True) -> pd.DataFrame:
        query = self.database_manager.database.session.query(ComplexComposition)
        complex_composition = pd.read_sql(query.statement, self.database_manager.database.engine)

        protein_gene_join = Protein.protein_multidata_id == Multidata.id_multidata
        if include_gene:
            gene_protein_join = Gene.protein_id == Protein.id_protein
            multidatas_proteins_query = self.database_manager.database.session.query(
                Gene, Protein, Multidata).join(Protein, gene_protein_join).join(Multidata, protein_gene_join)
        else:
            multidatas_proteins_query = self.database_manager.database.session.query(Protein, Multidata).join(
                Multidata, protein_gene_join)

        multidatas_proteins = pd.read_sql(multidatas_proteins_query.statement, self.database_manager.database.engine)
        multidatas_proteins.columns = multidatas_proteins.columns.map(lambda column: column + '_protein')

        multidatas_complexes_query = self.database_manager.database.session.query(Multidata)
        multidatas_complexes = pd.read_sql(multidatas_complexes_query.statement, self.database_manager.database.engine)
        multidatas_complexes.columns = multidatas_complexes.columns.map(lambda column: column + '_complex')

        complex_composition_expanded = pd.merge(complex_composition, multidatas_complexes,
                                                left_on='complex_multidata_id',
                                                right_on='id_multidata_complex')

        complex_composition_expanded = pd.merge(complex_composition_expanded, multidatas_proteins,
                                                left_on='protein_multidata_id',
                                                right_on='id_multidata_protein')

        return complex_composition_expanded

    def get_complex_by_multidatas(self, multidatas: pd.DataFrame, all_proteins_expressed: bool = True) -> pd.DataFrame:
        complex_composition = self.get_all_compositions()

        multidatas_ids = multidatas['id_multidata'].to_frame()
        complex_composition_merged = pd.merge(complex_composition, multidatas_ids, left_on='protein_multidata_id',
                                              right_on='id_multidata')  # 默认inner join

        if complex_composition_merged.empty:
            return complex_composition_merged

        def all_protein_expressed(complex):
            number_proteins_in_counts = len(
                complex_composition_merged[
                    complex_composition_merged['complex_multidata_id'] == complex['complex_multidata_id']])

            if number_proteins_in_counts < complex['total_protein']:
                return False

            return True

        if all_proteins_expressed:
            complex_composition_merged = complex_composition_merged[
                complex_composition_merged.apply(all_protein_expressed, axis=1)]

        complexes = self.get_all_expanded()
        complex_composition_merged = pd.merge(complex_composition_merged, complexes,
                                              left_on='complex_multidata_id',
                                              right_on='id_multidata',
                                              suffixes=['_protein', ''])

        complex_composition_merged.drop_duplicates(['complex_multidata_id'], inplace=True)

        return complex_composition_merged

    def add(self, complexes):
        """
        Uploads complex data from csv.

        - Creates new complexes in Multidata table
        - Creates reference in Complex table
        - Creates complex composition to define complexes.
        """

        if complexes.empty:
            return

        existing_complexes = self.database_manager.database.session.query(Multidata.name).all()
        existing_complexes = [c[0] for c in existing_complexes]
        proteins = self.database_manager.database.session.query(Multidata.name, Multidata.id_multidata).join(
            Protein).all()
        proteins = {p[0]: p[1] for p in proteins}

        # Get complex composition info
        complete_indices = []
        incomplete_indices = []
        missing_proteins = []
        complex_map = {}
        for index, row in complexes.iterrows():
            missing = False
            protein_id_list = []
            for protein in ['protein_1', 'protein_2',
                            'protein_3', 'protein_4']:
                if not pd.isnull(row[protein]):
                    protein_id = proteins.get(row[protein])
                    if protein_id is None:
                        missing = True
                        missing_proteins.append(row[protein])
                    else:
                        protein_id_list.append(protein_id)
            if not missing:
                complex_map[row['name']] = protein_id_list
                complete_indices.append(int(index))
            else:
                incomplete_indices.append(int(index))

        if len(incomplete_indices) > 0:
            logger.warning('MISSING PROTEINS:')
            for protein in missing_proteins:
                logger.warning('MISSING PROTEINS:')
                logger.warning(protein)

            logger.warning('COMEPLEXES WITH MISSING PROTEINS:')
            incomplete_complexes = complexes.iloc[incomplete_indices, :]['name']
            for incomplete_complex in incomplete_complexes:
                logger.warning(incomplete_complex)

        # Insert complexes
        if not complexes.empty:
            # Remove unwanted columns
            removal_columns = list(
                [x for x in complexes.columns if 'protein_' in x or 'Name_' in x or 'Unnamed' in x])
            # removal_columns += ['comments']
            complexes.drop(removal_columns, axis=1, inplace=True)

            # Remove rows with missing complexes
            complexes = complexes.iloc[complete_indices, :]

            # Convert ints to bool
            bools = ['receptor', 'other', 'secreted_highlight', 'transmembrane', 'secreted',
                     'peripheral']
            complexes[bools] = complexes[bools].astype(bool)

            # Drop existing complexes
            complexes = complexes[complexes['name'].apply(
                lambda x: x not in existing_complexes)]

            multidata_df = remove_not_defined_columns(complexes.copy(),
                                                      self.database_manager.get_column_table_names(
                                                          'multidata_table'))

            multidata_df = self._add_complex_optimitzations(multidata_df)
            multidata_df.to_sql(name='multidata_table', if_exists='append', con=self.database_manager.database.engine,
                                index=False, chunksize=50)

        # Now find id's of new complex rows
        new_complexes = self.database_manager.database.session.query(Multidata.name, Multidata.id_multidata).all()
        new_complexes = {c[0]: c[1] for c in new_complexes}

        # Build set of complexes
        complex_set = []
        complex_table = []
        for complex_name in complex_map:
            complex_id = new_complexes[complex_name]
            for protein_id in complex_map[complex_name]:
                complex_set.append((complex_id, protein_id, len(complex_map[complex_name])))
            complex_table.append({'complex_multidata_id': complex_id, 'name': complex_name})

        # Insert complex composition
        complex_set_df = pd.DataFrame(complex_set,
                                      columns=['complex_multidata_id', 'protein_multidata_id', 'total_protein'])

        complex_table_df = pd.DataFrame(complex_table)
        complex_table_df = pd.merge(complex_table_df, complexes, on='name')

        remove_not_defined_columns(complex_table_df,
                                   self.database_manager.get_column_table_names('complex_table'))

        complex_table_df.to_sql(
            name='complex_table', if_exists='append',
            con=self.database_manager.database.engine, index=False, chunksize=50)

        complex_set_df.to_sql(
            name='complex_composition_table', if_exists='append',
            con=self.database_manager.database.engine, index=False, chunksize=50)

    @staticmethod
    def _add_complex_optimitzations(multidatas):
        multidatas['is_complex'] = True

        return multidatas


class GeneRepository(Repository):
    name = 'gene'

    def get_all(self):
        query = self.database_manager.database.session.query(Gene)
        result = pd.read_sql(query.statement, self.database_manager.database.session.bind)

        return result

    def get_all_expanded(self):
        """
        Join gene, protein, multidata to get full gene information
        :return:
        """
        protein_multidata_join = Protein.protein_multidata_id == Multidata.id_multidata
        gene_protein_join = Gene.protein_id == Protein.id_protein
        query = self.database_manager.database.session.query(Gene, Protein, Multidata).join(
            Protein, gene_protein_join).join(Multidata, protein_multidata_join)

        result = pd.read_sql(query.statement, self.database_manager.database.session.bind)

        return result

    def add(self, genes: pd.DataFrame):
        query_multidatas = self.database_manager.database.session.query(Protein.id_protein, Multidata.name).join(
            Multidata)
        multidatas = pd.read_sql(query_multidatas.statement, self.database_manager.database.session.bind)

        genes = self._blend_multidata(genes, ['name'], multidatas)

        genes.rename(index=str, columns={'id_protein': 'protein_id'}, inplace=True)
        genes = remove_not_defined_columns(genes, self.database_manager.get_column_table_names('gene_table'))

        genes.to_sql(name='gene_table', if_exists='append', con=self.database_manager.database.engine, index=False,
                     chunksize=50)

    @staticmethod
    def _blend_multidata(original_data: pd.DataFrame, original_column_names: list, multidatas: pd.DataFrame,
                         quiet: bool = False) -> pd.DataFrame:
        """
        Merges dataframe with multidata names in multidata ids
        """
        if quiet:
            logger.debug('Blending proteins in quiet mode')

        result = GeneRepository.blend_dataframes(original_data, original_column_names, multidatas, 'name', 'multidata')

        return result


class InteractionRepository(Repository):
    name = 'interaction'

    def get_all(self):
        """
        Get all data in the table
        :return: pd.DataFrame
        """
        query = self.database_manager.database.session.query(Interaction)
        interactions = pd.read_sql(query.statement, self.database_manager.database.engine)

        return interactions

    def get_interactions_by_multidata_id(self, id):
        """
        Filter out the interactions containing given multidata ID
        :type id: int
        :rtype: pd.DataFrame
        """
        query = self.database_manager.database.session.query(Interaction).filter(
            or_(Interaction.multidata_1_id == int(id), Interaction.multidata_2_id == int(id)))
        result = pd.read_sql(query.statement, self.database_manager.database.engine)

        return result

    def get_interactions_multidata_by_multidata_id(self, id):
        """

        :type id: int
        :rtype: pd.DataFrame
        """

        interactions = self.get_interactions_by_multidata_id(id)
        multidatas_expanded = self.database_manager.get_repository('multidata').get_all_expanded()
        interactions_expanded = self.expand_interactions_multidatas(interactions, multidatas_expanded)
        return interactions_expanded

    def get_all_expanded(self, include_gene=True, suffixes=('_1', '_2')):
        interactions_query = self.database_manager.database.session.query(Interaction)

        interactions = pd.read_sql(interactions_query.statement, self.database_manager.database.engine)

        multidata_expanded: pd.DataFrame = self.database_manager.get_repository('multidata').get_all_expanded(
            include_gene)

        multidata_expanded = multidata_expanded.astype({'id_multidata': 'int64'})

        interactions = pd.merge(interactions, multidata_expanded, left_on=['multidata_1_id'], right_on=['id_multidata'])
        interactions = pd.merge(interactions, multidata_expanded, left_on=['multidata_2_id'], right_on=['id_multidata'],
                                suffixes=suffixes)

        return interactions

    def add(self, interactions):
        interaction_df = self.blend_dataframes(interactions, ['partner_a', 'partner_b'],
                                               self.database_manager.get_repository('multidata').get_all_name_id(),
                                               'name', 'multidata')

        remove_not_defined_columns(interaction_df,
                                   self.database_manager.get_column_table_names('interaction_table'))

        interaction_df.to_sql(name='interaction_table', if_exists='append', con=self.database_manager.database.engine,
                              index=False, chunksize=50)

    @staticmethod
    def expand_interactions_multidatas(interactions: pd.DataFrame, multidatas_expanded,
                                       suffixes: list = ('_1', '_2')) -> pd.DataFrame:
        interactions_expanded = pd.merge(interactions, multidatas_expanded, left_on='multidata_1_id',
                                         right_on='id_multidata')

        interactions_expanded = pd.merge(interactions_expanded, multidatas_expanded, left_on='multidata_2_id',
                                         right_on='id_multidata', suffixes=suffixes)

        interactions_expanded.drop_duplicates(inplace=True)
        return interactions_expanded


class MultidataRepository(Repository):
    name = 'multidata'

    def get_all(self):
        """
        1540 rows
        :return:
        """
        query = self.database_manager.database.session.query(Multidata)
        result = pd.read_sql(query.statement, self.database_manager.database.engine)

        return result

    def get_all_expanded(self, include_gene=True):
        protein_multidata_join = Protein.protein_multidata_id == Multidata.id_multidata
        if include_gene:
            gene_protein_join = Gene.protein_id == Protein.id_protein
            query_single = self.database_manager.database.session.query(Gene, Protein, Multidata)
            query_single = query_single.join(Protein, gene_protein_join).join(Multidata, protein_multidata_join)
        else:
            query_single = self.database_manager.database.session.query(Protein, Multidata).join(
                Multidata, protein_multidata_join)

        multidata_simple = pd.read_sql(query_single.statement, self.database_manager.database.engine)

        multidata_complex_join = Multidata.id_multidata == Complex.complex_multidata_id
        query_complex = self.database_manager.database.session.query(Multidata, Complex).join(
            Complex, multidata_complex_join)
        multidata_complex = pd.read_sql(query_complex.statement, self.database_manager.database.engine)

        if multidata_complex.empty:
            return multidata_simple

        multidata_expanded = multidata_simple.append(multidata_complex, ignore_index=True, sort=True)

        return multidata_expanded

    def get_all_name_id(self) -> pd.DataFrame:
        query_multidatas = self.database_manager.database.session.query(Multidata.id_multidata, Multidata.name)
        multidatas = pd.read_sql(query_multidatas.statement, self.database_manager.database.session.bind)

        return multidatas

    def get_multidatas_from_string(self, input_string: str) -> pd.DataFrame:
        multidatas = self.get_all_expanded()

        return multidatas[(multidatas['name'] == input_string) |
                          (multidatas['ensembl'] == input_string) |
                          (multidatas['protein_name'] == input_string) |
                          (multidatas['gene_name'] == input_string)]


class ProteinRepository(Repository):
    name = 'protein'

    def get_all(self) -> pd.DataFrame:
        """
        1281 rows
        :return:
        """
        protein_query = self.database_manager.database.session.query(Protein)
        protein = pd.read_sql(protein_query.statement, self.database_manager.database.session.bind)

        return protein

    def get_all_expanded(self) -> pd.DataFrame:
        """
        Expand protein information from multidata（inner join）
        :return:
        """
        protein_multidata_join = Protein.protein_multidata_id == Multidata.id_multidata
        protein_query = self.database_manager.database.session.query(Protein, Multidata).join(
            Multidata, protein_multidata_join)
        protein = pd.read_sql(protein_query.statement, self.database_manager.database.session.bind)

        return protein

    def get_all_name_id(self) -> pd.DataFrame:
        query_multidatas = self.database_manager.database.session.query(Protein.id_protein, Multidata.name).join(
            Multidata)
        multidatas = pd.read_sql(query_multidatas.statement, self.database_manager.database.session.bind)

        return multidatas

    def get_protein_multidata_by_uniprot(self, uniprot: str) -> pd.DataFrame:
        protein_query = self.database_manager.database.session.query(Protein, Multidata).join(Multidata).filter_by(
            name=uniprot).limit(1)
        protein = pd.read_sql(protein_query.statement, self.database_manager.database.session.bind)

        if not protein.empty:
            return protein.iloc[0, :]
        return None

    def add_proteins(self, proteins: pd.DataFrame, multidatas: pd.DataFrame):
        multidatas.to_sql(name='multidata_table', if_exists='append', con=self.database_manager.database.engine,
                          index=False, chunksize=50)

        multidata_query = self.database_manager.database.session.query(Multidata.id_multidata, Multidata.name)
        multidatas_db = pd.read_sql(multidata_query.statement, self.database_manager.database.session.bind)
        multidatas_db.rename(index=str, columns={'id_multidata': 'protein_multidata_id'}, inplace=True)
        proteins_to_add = pd.merge(proteins, multidatas_db, on='name')
        proteins_to_add.drop('name', inplace=True, axis=1)

        proteins_to_add.to_sql(name='protein_table', if_exists='append', con=self.database_manager.database.engine,
                               index=False, chunksize=50)
