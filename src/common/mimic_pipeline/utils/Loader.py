import pandas as pd
import dask.dataframe as dd
from sqlalchemy import create_engine
from abc import ABC, abstractmethod

class AbstractLoader():
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __getitem__(self):
        pass

class Loader(AbstractLoader):
    """
    for helping with reading data from folders
    
    Parameters
    ----------
    path (str): 
        the path to which the dataset is stored, default to path in cluster "/usr/xtmp/mimic2023/mimic3"
    
    Usage Example
    -------------
    >>> loader = ClusterLoader()    # specify the path if you are running locally (different from cluster)
    >>> df = loader['CPTEVENTS']    # to obtain CPTEVENTS.csv table
    >>> df1, df2 = loader['ADMISSIONS', 'CPTEVENTS']    # to obtain ADMISSIONS.csv and CPTEVENTS.csv, respectively
    >>> # you can do as many as you want (but be careful with memory usage)
    >>> df1, df2, df3, df4 = loader['ADMISSIONS', 'CPTEVENTS', 'NOTEEVENTS', 'CALLOUT']
    >>> # or as a list
    >>> df_list = loader['ADMISSIONS', 'CPTEVENTS', 'NOTEEVENTS', 'CALLOUT']
    >>> # change mode
    >>> loader.mode('dd')    # change to dask.dataframe (fast for large datasets)
    >>> df = loader['CHARTEVENTS']  # load CHARTEVENTS using dask
    >>> # to check mode
    >>> print(loader.cur_mode)
    >>> 'dd'
    """
    def __init__(self, path: str='/usr/xtmp/mimic2023/mimic3', mode: str='pd', format_cols='lower') -> None:
        assert type(path) == str, 'invalid path'
        self.path = path
        self.cur_mode = mode
        self.columns_format = format_cols
        self.mode = mode

    def __getitem__(self, items):
        return self.get_tables(items)
    
    def get_mode(self):
        if self.cur_mode == "dd":
            return dd
        elif self.cur_mode == "pd":
            return pd
        else: 
            raise ValueError("Invalid mode " + self.cur_mode)
    
    def get_csv(self, table_name):
        lib = self.get_mode()
        df = lib.read_csv(f"{self.path}/{table_name}.csv", dtype="object")
        if self.columns_format == "upper":
            df.columns = df.columns.str.upper()
        elif self.columns_format == "lower":
            df.columns = df.columns.str.lower()
        else:
            raise ValueError("Invalid column format " + self.columns_format)
        return df
    
    def get_tables(self, *args):
        if isinstance(args[0], str):
            return self.get_csv(args[0])
        else:
            return [self.get_csv(table_name) for table_name in args[0]]
    
    def set_mode(self, mode: str):
        """
        To alter mode of loading
        Args:
            mode (str): must be in {'pd', 'dd'}, which stands for switching to reading with pandas or dask
        """
        assert mode in ['pd', 'dd'], "mode must be 'pd' or 'dd'"
        self.cur_mode = mode


class DataBaseLoader():
    """
    For querying from PostgresSQL local database (requires setting up the local database first, see link below)
    https://github.com/jack-y-xu/mimic-pipeline/tree/main/mimic-code

    Parameters
    ----------
    user : str
    password : str
    dbname : str, optional
        database name, by default "mimic"
    port : str, optional
        port you are using, by default "5432"
    schema : str, optional
        schema you are using, by default "mimiciii", if you are not using schema, set it to None
    host : str, optional
        host you are on, by default "localhost"
    
    Example
    -------
    >>> dbloader = DataBaseLoader(
        user="mt361",
        password="InterpretableML-is-Fun",
    )
    >>> data = dbloader["sapsii"]       # query sapsii table
    >>> data2 = dbloader.query("SELECT count(icustay_id) FROM sapsii;")     # making query
    """
    def __init__(self, user:str, password:str, dbname:str="mimic", port:str="5432", schema:str="mimiciii", host:str="localhost") -> None:
        assert all(isinstance(e, str) for e in [dbname, user, password, host, port]), "those parameters must be string!"
        assert isinstance(schema, str) or isinstance(schema, None), "\"schema\" must be str or None!"
        self.dbname = dbname
        if schema is not None:
            self.engine = create_engine(
                f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}",
                connect_args={'options': f"-c search_path={schema}"}
            )
        else:
            self.engine = create_engine(
                f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
            )
    
    def __getitem__(self, TableName:str) -> pd.DataFrame:
        assert isinstance(TableName, str), "\"TableName\" must be a string!"
        table = pd.read_sql_query(f"SELECT * FROM {TableName}", self.engine)
        
        return table
    
    def query(self, command:str) -> pd.DataFrame:
        '''for making query from database, with arbitrary SQL command'''
        assert isinstance(command, str), "\"command\" must be a string!"
        query = pd.read_sql_query(command, self.engine)
        
        return query