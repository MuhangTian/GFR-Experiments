from dataclasses import dataclass

@dataclass
class BaseDataClass:

    def all(self): return list(self.__annotations__.keys())


@dataclass
class FeatureNaNPercentage(BaseDataClass):  # this is NaN percentage for features in 24 hours period, NOT BINNED
    aado2: float = 0.94
    albumin_max: float = 0.63
    albumin_min: float = 0.63
    bilirubin_max: float = 0.56
    bilirubin_min: float = 0.56
    paco2: float = 0.33
    pao2: float = 0.46
    pao2fio2_novent_min: float = 0.73
    pao2fio2_vent_min: float = 0.58
    ph: float = 0.33
    rate_dobutamine: float = 0.99
    rate_dopamine: float = 0.95
    rate_epinephrine: float = 0.96
    rate_norepinephrine: float = 0.89


@dataclass
class OASISFeatures(BaseDataClass):
    """
    NOTE: unconventional use: variable: actual_data_type = "description | unit of measurement"
    """
    gcs:int = "Glascow Coma Score | integers from 3 to 15"
    age:float = "years"
    preiculos:float = "length of stay in minutes"
    
    heartrate_min:int = "beats per minute"
    heartrate_max:int = "beats per minute"
    heartrate_mean:int = 'beats per minute'
    
    resprate_min:int = "breaths per minute"
    resprate_max:int = "breaths per minute"
    resprate_mean:int = 'breaths per minute'
    
    meanbp_min:float = "Min Mean Arterial Pressure | mmHg (milimeters of mercury)"      # MAP = 1/3 * SBP + 2/3 * DBP (diastolic pressure)
    meanbp_max:float = "Max Mean Arterial Pressure | mmHg (milimeters of mercury)"
    meanbp_mean:float = 'same as above'
    
    electivesurgery:int = "yes or no"
    mechvent:int = "yes or no"
    urineoutput:float = "cubic centimeter per day"
    
    tempc_min:float = "degrees celsius"
    tempc_max:float = "degrees celsius"
    tempc_mean:float = 'degrees celsius'
    
@dataclass
class OASISRawFeatures(BaseDataClass):
    """
    NOTE: unconventional use: variable: actual_data_type = "description | unit of measurement"
    """
    gcs:int = "Glascow Coma Score | integers from 3 to 15"
    age:float = "years"
    preiculos:float = "length of stay in minutes"
    heartrate:int = "beats per minute"
    resprate:int = "breaths per minute"
    meanbp:float = "Min Mean Arterial Pressure | mmHg (milimeters of mercury)"      # MAP = 1/3 * SBP + 2/3 * DBP (diastolic pressure)
    electivesurgery:int = "yes or no"
    mechvent:int = "yes or no"
    urineoutput:float = "cubic centimeter per day"
    tempc:float = "degrees celsius"
    

@dataclass
class SAPSIIFeatures(BaseDataClass): # NOTE: unconventional use: variable: actual_data_type = "unit of measurement"
    """
    NOTE: unconventional use: variable: actual_data_type = "description | unit of measurement"
    """
    sysbp_min:int = "Min Systolic Blood Pressure | mmHg"
    sysbp_max:int = "Max Systolic Blood Pressure | mmHg"
    sysbp_mean:int = 'same as above'
    # Systolic: when the heart contracts and pumps, diastolic: when the heart relax
    
    pao2fio2_vent_min:float = "Minimum PaO2/FiO2 ratio for ventilated patients | ratio"     
    # PaO2: Partial pressure of oxygen in arterial blood, aka oxygen in your blood
    # FiO2: inspiratory oxygen concentration, aka fraction of inspired oxygen
    
    bun_max:int = "Max Blood Urea Nitrogen | mg/dL (milligrams per deciliter)"
    bun_min:int = "Min Blood Urea Nitrogen | mg/dL (milligrams per deciliter)"
    bun_mean:int = 'same as above'
    # indicator of kidney health, deciliter is a tenth of a liter
    
    wbc_min:int = "Min White Blood Cell Count | thousands per microliter"
    wbc_max:int = "Max White Blood Cell Count | thousands per microliter"
    wbc_mean:int = 'same as above'
    
    potassium_min:float = "Min Potassium in Blood Serum | mEq/L (Miliequivalents per liter)"
    potassium_max:float = "Max Potassium in Blood Serum | mEq/L (Miliequivalents per liter)"
    potassium_mean:float = 'same as above'
    # An equivalent is the amount of a substance that will react with a certain number of hydrogen ions. A milliequivalent is one-thousandth of an equivalent.
    
    sodium_min:int = "Min Sodium in Blood | mEq/L (Miliequivalents per liter)"
    sodium_max:int = "Max Sodium in Blood | mEq/L (Miliequivalents per liter)"
    sodium_mean:int = 'same as above'
    # indicator of kidney function and hydration
    
    bicarbonate_min:int = "Min Serum Bicarbonate | mEq/L"
    bicarbonate_max:int = "Max Serum Bicarbonate | mEq/L"
    bicarbonate_mean:int = 'same as above'
    # indicator whether body is able to maintain acid/base balance
    
    bilirubin_min:float = "Min Bilirubin in Blood | mg/dL (milligrams per deciliter)"
    bilirubin_max:float = "Max Bilirubin in Blood | mg/dL (milligrams per deciliter)"
    bilirubin_mean:float = 'same as above'
    # indicator of kidney function, high value of this number means your liver is not doing well

@dataclass
class SAPSIIRawFeatures(BaseDataClass): # NOTE: unconventional use: variable: actual_data_type = "unit of measurement"
    """
    NOTE: unconventional use: variable: actual_data_type = "description | unit of measurement"
    """
    sysbp:int = "Min Systolic Blood Pressure | mmHg"
    pao2fio2_vent_min:float = "Minimum PaO2/FiO2 ratio for ventilated patients | ratio"     
    bun:int = "Max Blood Urea Nitrogen | mg/dL (milligrams per deciliter)"
    wbc:int = "Min White Blood Cell Count | thousands per microliter"
    potassium:float = "Min Potassium in Blood Serum | mEq/L (Miliequivalents per liter)"
    sodium:int = "Min Sodium in Blood | mEq/L (Miliequivalents per liter)"
    bicarbonate:int = "Min Serum Bicarbonate | mEq/L"
    bilirubin:float = "Min Bilirubin in Blood | mg/dL (milligrams per deciliter)"
    

@dataclass
class APSIIIFeatures(BaseDataClass):
    """
    NOTE: unconventional use: variable: actual_data_type = "description | unit of measurement"
    """
    pao2_max:int = "Arterial Oxygen Partial Pressure | mmHg"
    aado2_max:int = "Alveolar Arterial Oxygen Gradient | mmHg"      
    # aado2 = alveolar oxygen partial pressure - arterial oxygen partial pressure
    
    ph_max:float = "pH Level | [1, 14]"
    paco2_max:int = "Arterial Carbon Dioxide Partial Pressure | mmHg"
    
    glucose_min:int = "Min Blood Glucose | mg/dL"
    glucose_max:int = "Max Blood Glucose | mg/dL"
    glucose_mean:int = 'same as above'
    
    hematocrit_min:float = "Min Hematocrit | proportion of RBCs in blood %"
    hematocrit_max:float = "Max Hematocrit | proportion of RBCs in blood %"
    hematocrit_mean:float = 'same as above'
    
    creatinine_min:float = "Min Creatinine | mg/dL"
    creatinine_max:float = "Max Creatinine | mg/dL"
    creatinine_mean:float = 'same as above'
    # creatinine is a waste product that comes from wear/tear of muscles in body, another measure of kidney function
    
    albumin_min:float = "Min Albumin | g/dL"
    albumin_max:float = "Max Albumin | g/dL"
    albumin_mean:float = 'same as above'
    # albumin: a protein in blood plasma, this measure can allow doctors to assess kidney function
    
@dataclass
class APSIIIRawFeatures(BaseDataClass):
    """
    NOTE: unconventional use: variable: actual_data_type = "description | unit of measurement"
    """
    pao2:int = "Arterial Oxygen Partial Pressure | mmHg"
    aado2:int = "Alveolar Arterial Oxygen Gradient | mmHg"      
    ph:float = "pH Level | [1, 14]"
    paco2:int = "Arterial Carbon Dioxide Partial Pressure | mmHg"
    glucose:int = "Min Blood Glucose | mg/dL"
    hematocrit:float = "Min Hematocrit | proportion of RBCs in blood %"
    creatinine:float = "Min Creatinine | mg/dL"
    albumin:float = "Min Albumin | g/dL"

@dataclass
class SOFAFeatures(BaseDataClass):
    """
    NOTE: unconventional use: variable: actual_data_type = "description | unit of measurement"
    """
    rate_dobutamine_max:float = 'max rate of injection of dobutamine | micrograms/kg/minute'
    # dobutamine is used to manage low blood pressure, requires close monitoring since too much cause high blood pressure
    rate_epinephrine_max:float = 'max rate of injection of epinephrine | micrograms/kg/minute'
    # epinephrine is another name for adrenaline, used for life-threatening allergic reactions, such as heart stop pumping or coma
    rate_dopamine_max:float = 'max rate of injection of dopamine | micrograms/kg/minute'
    # dopamine is used to treat low blood pressure, coma, and kidney failure
    rate_norepinephrine_max:float = 'max rate of injection of norepinephrine | micrograms/kg/minute'
    # norepinephrine is used to raise blood pressure
    platelet_min:float = 'minimum platelet count | thousands per microliter of blood'
    platelet_max:float = 'maximum platelet count | thousands per microliter of blood'

@dataclass
class SOFAFeatures(BaseDataClass):
    """
    NOTE: unconventional use: variable: actual_data_type = "description | unit of measurement"
    """
    rate_dobutamine:float = 'max rate of injection of dobutamine | micrograms/kg/minute'
    rate_epinephrine:float = 'max rate of injection of epinephrine | micrograms/kg/minute'
    rate_dopamine:float = 'max rate of injection of dopamine | micrograms/kg/minute'
    rate_norepinephrine:float = 'max rate of injection of norepinephrine | micrograms/kg/minute'
    platelet:float = 'minimum platelet count | thousands per microliter of blood'

@dataclass
class SelectedFeatures(BaseDataClass):
    '''To keep track of selected features, type is only a placeholder doesn't matter here, date: 4.4 '''
    urineoutput: float = ''
    pao2fio2_vent_min: float = ''
    mechvent: float = ''
    age: float = ''
    bun_max: float = ''
    bun_min: float = ''
    resprate_min: float = ''
    mechvent_pct: float = ''
    gcs: float = ''
    aado2: float = ''
    sysbp_min: float = ''
    electivesurgery: float = ''
    tempc_max: float = ''
    glucose_min: float = ''