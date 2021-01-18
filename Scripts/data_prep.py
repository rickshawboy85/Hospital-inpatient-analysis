#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:09:33 2021

@author: samthomas
"""



import numpy as np


def prep_dataset(hospital):
    
    print('Cleaning data. Process may take a minute or two...')
    
    hospital.drop_duplicates(keep='first', inplace=True)

    # drop irrelevant columns
    hospital.drop(columns=['Total Costs', 
                           'Total Charges', 
                           'Birth Weight',
                           'Race',
                           'Ethnicity',
                           'Gender',
                           'Other Provider License Number', 
                           'Operating Provider License Number', 
                           'Attending Provider License Number', 
                           'Payment Typology 3', 
                           'Payment Typology 2', 
                           'Payment Typology 1', 
                           'Discharge Year', 
                           'Patient Disposition', 
                           'Zip Code - 3 digits', 
                           'Facility Name', 
                           'Facility Id', 
                           'Operating Certificate Number', 
                           'Hospital County', 
                           'Health Service Area',
                           'APR MDC Code',
                           'CCS Diagnosis Code',
                           'CCS Procedure Code',
                           'APR DRG Code',
                           'APR Severity of Illness Code',
                           'Emergency Department Indicator',
                           'CCS Diagnosis Description',
                           'APR Medical Surgical Description'], axis=1, inplace=True)


    # remove punctuation
    hospital['Length of Stay'].replace(['120+', '120 +'], '120', inplace=True)
    
    hospital['Length of Stay'] = hospital['Length of Stay'].astype(int)
    

    # combine age groups with similar LoS averages
    hospital['Age Group'] = hospital['Age Group'].replace('18 to 29', '18 to 49').replace('30 to 49', '18 to 49')
    hospital['Age Group'] = hospital['Age Group'].replace('50 to 69', '50 or Older').replace('70 or Older', '50 or Older')
    

    # impute missings 
    hospital['APR Severity of Illness Description'] = hospital['APR Severity of Illness Description'].fillna('Minor')

        
    # impute missings
    hospital['APR Risk of Mortality'] = hospital['APR Risk of Mortality'].fillna('Minor') 

    
    ### reduce categories in the major diagnostic categories variable
    # pregnancy
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Pregnancy|Newborns'), 'Pregnancy/Neonates', hospital['APR MDC Description'])
   
    # blood/cardio
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Blood|Circulatory'), 'Blood/Circulatory System', hospital['APR MDC Description'])
        
    # skin/muscle/skeletal
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Burns|Musculoskeletal|Skin'), 'Skin/Muscle/Skeletal', hospital['APR MDC Description'])
   
    # Liver/Kidneys
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Hepatobiliary|Kidney'), 'Liver/Kidneys/Urinary', hospital['APR MDC Description'])   
    
    # endocrine/digestive
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Digestive|Endocrine'), 'Endocrine/Metabolic/Gastro', hospital['APR MDC Description'])
    
    # male/female reproduction
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Reproductive'), 'Male/Female Reproductive System', hospital['APR MDC Description'])
    
    # Mental disorders
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Mental'), 'Mental Disorders', hospital['APR MDC Description'])
    
    # infectious disease
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Infectious|Immunodeficiency'), 'Infectious Disease', hospital['APR MDC Description'])
   
    # trauma and rehab
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Trauma|Rehabilitation'), 'Trauma/Rehabilitation', hospital['APR MDC Description'])
   
    # eyes/craniofacial
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Eye|Craniofacial'), 'Eyes/Craniofacial', hospital['APR MDC Description'])
    
    # Lymphatic/Chemotherapy
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Lymphatic'), 'Lymphatic/Chemo/Radiotherapy', hospital['APR MDC Description'])
    
    # Poisonings/Other Complications
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Poisonings|Ungroupable'), 'Poisonings/Other Complications', hospital['APR MDC Description'])
   
    # Respiratory
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Respiratory'), 'Blood/Circulatory System', hospital['APR MDC Description'])
    
    # Nervous System
    hospital['APR MDC Description'] = np.where(hospital['APR MDC Description'].str.contains('Nervous|Mental'), 'Mental/Nervous Disorders', hospital['APR MDC Description'])
    
    
    
    ### group procedures by length of stay. longer LoS = more invasive.
    hospital['CCS Procedure Description'] = np.where(hospital['Length of Stay'] < 2, 'Non-invasive', hospital['CCS Procedure Description'])

    hospital['CCS Procedure Description'] = np.where((hospital['Length of Stay'] >= 2) &  (hospital['Length of Stay'] < 3), 'Slightly invasive', hospital['CCS Procedure Description'])

    hospital['CCS Procedure Description'] = np.where((hospital['Length of Stay'] >= 3) &  (hospital['Length of Stay'] < 5), 'Moderately invasive', hospital['CCS Procedure Description'])

    hospital['CCS Procedure Description'] = np.where((hospital['Length of Stay'] >= 5) &  (hospital['Length of Stay'] < 10), 'Invasive', hospital['CCS Procedure Description'])

    hospital['CCS Procedure Description'] = np.where((hospital['Length of Stay'] >= 10) &  (hospital['Length of Stay'] < 20), 'Highly invasive or chronic condition', hospital['CCS Procedure Description'])

    hospital['CCS Procedure Description'] = np.where(hospital['Length of Stay'] >= 20, 'Treatment for chronic condition', hospital['CCS Procedure Description'])
    
    
    # ### reduce categories in 'Type of Admission' variable
    hospital['Type of Admission'].replace('Not Available', 'Newborn', inplace=True)
    hospital['Type of Admission'].replace('Urgent', 'Emergency', inplace=True)
    
    
    # ### create new variables from diagnosis group descriptions
    hospital['APR DRG Description'] = hospital['APR DRG Description'].map(lambda x: x.lower())
    
    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('neonate|delivery|antepartum|Uterine|Postpartum|labor|obstetric|pregnancy|curettage'), 'Births', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('infections|infection|infectious|viral|hiv|immunologic|Fever|allergic|fever|parasitic'), 'Infectios disease', hospital['APR DRG Description'])
    
    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('heart|cardiac|endocarditis|infarct|coronary|syncope|myocardial|angina|chest|blood|vascular|hypovolemia|anemia|ischemia|hypertension|circulatory|platelet|lymphatic|cardiothoracic|ventricular|precerebral|cardiomyopathy|splenectomy'), 'Heart/circulatory_', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('pulmonary|pneumonia|asthma|respiratory|tracheostomy|lung|trachea'), 'Respiratory', hospital['APR DRG Description'])
    
    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('schizophrenia|bipolar|depressive|mental|anxiety|behavioral|personality'), 'Mental disorders', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('skin|burns'), 'Skin', hospital['APR DRG Description'])
    
    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('urinary|genitourinary|bladder|kidney|renal|nephritis'), 'Kidney and urinary', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('urethral|male|transurethral|scrotal|penis'), 'Male genital', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('seizure|nervous|nerve|craniotomy|migraine|nervous|demyelinating'), 'Neural', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('rehabilitation|aftercare'), 'Rehabilitation', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('poisoning|poisonings|toxic|rehabilitation|alcohol|dependence|stupor'), 'Substance abuse and poisoning/toxicity', hospital['APR DRG Description'])
    
    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('gastroenteritis|bowel|digestive|diverticulitis|intestinal|obesity|gastrointestinal|laparoscopic|appendectomy|gastritis|biliary|abdominal|esophageal|nutritional|peritoneal|Anal|Eating'), 'Gastrointestinal_', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('diabetes|pancreas|liver'), 'Liver/Pancreas', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('musculoskeletal|lumbar|back|fracture|knee|joint|hip|disc|hernia|foot|connective|pelvic|pelvis|neck|amputation|tendon|spinal|femur|orthopedic|wrist'), 'Musculo skeletal', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('chemotherapy|leukemia|marrow|radiotherapy'), 'Cancer', hospital['APR DRG Description'])
    
    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('trauma|contusion/laceration'), 'Trauma', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('mastectomy|menstrual|female|breast|uterine'), 'Female', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('intracranial|cranial|head'), 'Head', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('complications|complication'), 'Complications', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('endocrine|thyroid|pituitary|metabolism'), 'Endocrine', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('ear|nose|throat|oral|tonsil|sinus|palate|labyrinth'), 'ENT', hospital['APR DRG Description'])
   
    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('eye|orbital'), 'Eyes', hospital['APR DRG Description'])

    hospital['APR DRG Description'] = np.where(hospital['APR DRG Description'].str.contains('ungroupable|invalid|signs'), 'Ungroupable', hospital['APR DRG Description'])
    

    # # delete rows with nan values
    hospital = hospital.dropna()
    
    
    print(hospital.head(), '\n\nDataframe dimensions: ', hospital.shape)
    
    
    return hospital
    
if __name__ == '__main__':    
    prep_dataset2(hospital)
    
    
    
    
    













