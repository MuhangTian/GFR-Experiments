import pandas as pd
import numpy as np

from mimic_pipeline.data import BASE_COLS

class Table:
    """
    To allow easier operations to use for element wise addition, subtraction, multiplication and division on the tables,
    while keeping the subject_ids same.
    
    Example
    -------
    >>> a, b = Table(a), Table(b)
    >>> # to get percentage change
    >>> change = (a-b)/b*100
    """
    def __init__(self, df) -> None:
        self.df = df
    
    def __add__(self, other):
        if isinstance(other, Table):
            self.__check_validity(other)
            df = self.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            other = other.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            result = np.add(df, other)
            result = pd.DataFrame(result, columns=list(self.columns))
            for i in range(len(BASE_COLS)):
                result.insert(i, BASE_COLS[i], self.base_col_df[BASE_COLS[i]])
        elif isinstance(other, float) or isinstance(other, int):
            self.__prepare()
            df = self.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            result = np.add(df, other)
            result = pd.DataFrame(result, columns=list(self.columns))
            for i in range(len(BASE_COLS)):
                result.insert(i, BASE_COLS[i], self.base_col_df[BASE_COLS[i]])
        else:
            raise ValueError("Type not supported!")
        
        return Table(result)
    
    def __sub__(self, other):
        if isinstance(other, Table):
            self.__check_validity(other)
            df = self.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            other = other.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            result = np.subtract(df, other)
            result = pd.DataFrame(result, columns=list(self.columns))
            for i in range(len(BASE_COLS)):
                result.insert(i, BASE_COLS[i], self.base_col_df[BASE_COLS[i]])
        elif isinstance(other, float) or isinstance(other, int):
            self.__prepare()
            df = self.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            result = np.subtract(df, other)
            result = pd.DataFrame(result, columns=list(self.columns))
            for i in range(len(BASE_COLS)):
                result.insert(i, BASE_COLS[i], self.base_col_df[BASE_COLS[i]])
        else:
            raise ValueError("Type not supported!")
        
        return Table(result)
    
    def __truediv__(self, other):
        if isinstance(other, Table):
            self.__check_validity(other)
            df = self.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            other = other.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            result = np.divide(df, other)       # element wise operation
            result = pd.DataFrame(result, columns=list(self.columns))
            for i in range(len(BASE_COLS)):
                result.insert(i, BASE_COLS[i], self.base_col_df[BASE_COLS[i]])
        elif isinstance(other, float) or isinstance(other, int):
            self.__prepare()
            df = self.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            result = np.divide(df, other)
            result = pd.DataFrame(result, columns=list(self.columns))
            for i in range(len(BASE_COLS)):
                result.insert(i, BASE_COLS[i], self.base_col_df[BASE_COLS[i]])
        else:
            raise ValueError("Type not supported!")
        
        return Table(result)
    
    def __mul__(self, other):
        if isinstance(other, Table):
            self.__check_validity(other)
            df = self.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            other = other.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            result = np.multiply(df, other)     # element wise operation
            result = pd.DataFrame(result, columns=list(self.columns))
            for i in range(len(BASE_COLS)):
                result.insert(i, BASE_COLS[i], self.base_col_df[BASE_COLS[i]])
        elif isinstance(other, float) or isinstance(other, int):
            self.__prepare()
            df = self.df.drop(BASE_COLS, axis=1).to_numpy().astype(float)
            result = np.multiply(df, other)
            result = pd.DataFrame(result, columns=list(self.columns))
            for i in range(len(BASE_COLS)):
                result.insert(i, BASE_COLS[i], self.base_col_df[BASE_COLS[i]])
        else:
            raise ValueError("Type not supported!")
        
        return Table(result)
    
    def __repr__(self): return str(self.df)
    
    def __len__(self): return len(self.df)
    
    def __check_validity(self, other):
        assert isinstance(self.df, pd.DataFrame) and isinstance(other.df, pd.DataFrame), "both must be pd.DataFrame!"
        assert list(self.df.columns) == list(other.df.columns), "column names don't match!"
        assert self.df.shape == other.df.shape, "shapes don't match!"
        pd.testing.assert_frame_equal(self.df[BASE_COLS], other.df[BASE_COLS], check_dtype=False), "subject_ids don't match!"
        self.base_col_df= self.df[BASE_COLS]
        self.columns = list(self.df.drop(BASE_COLS, axis=1).columns)
    
    def __prepare(self):
        self.base_col_df= self.df[BASE_COLS]
        self.columns = list(self.df.drop(BASE_COLS, axis=1).columns)
        
        
    
    

    
        