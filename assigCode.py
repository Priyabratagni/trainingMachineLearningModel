import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def getData(filename):
  canData=[] #List to store CAN data
  f = open(filename)
  #read_file = reader(f)
  read_file =f.readlines()
  
  #file = list(read_file)
  speed = []
  rpm = []
  i = 0
  for row in read_file:
    #Change the positions of the values if needed
    record = {'stamp':row[1:18], 'PID':row[25:28], 'const1':row[29:33], 
'change':row[33:41],'value':int(row[41:45], 16), 'value2':0 ,'attack':0}
    
    if record["PID"] == '254': #Processing of speed
      if record["value"] >= 4095:
        record["attack"] = 1
      record['value'] =  (record['value'] * 0.62137119) /100
      speed.append(record['value'])
    
    if record["PID"] == '115': #Processing of RPM 
      if record["value"] >= 65535:
        record["attack"] = 1
      record['value'] =  (record['value'] * 2)
      rpm.append(record['value'])

    i = i+1   
    canData.append(record)
    record={}
    
  f.close()
  return canData

def dict_to_df(dict):

  #load dictionary to dataframe
  df = pd.DataFrame.from_dict(dict)
  df = df.drop(columns=['stamp','const1','change','value2'])
  df = df.loc[(df['PID'] == '115') | (df['PID'] == '254')]
  df = df.reset_index(drop=True)
  one_hot = pd.get_dummies(df['PID'])
  df = df.drop('PID',axis = 1)
  df = df.join(one_hot)
  df = df[['115', '254', 'value', 'attack']]
  df.rename(columns = {'115':'RPM', '254':'Speed'}, inplace = True)

  # Intialize df as a object
  df = df.astype(object)

  df.loc[df['RPM'] == 1, 'RPM'] = df['value']
  df.loc[df['Speed'] == 1, 'Speed'] = df['value']
  df = df.drop(columns=['value'])

  return df

#edit file name with file directory of downloaded log files
fff_injection_df = dict_to_df(getData("speedReading.log"))
rpm_injection_df = dict_to_df(getData("rpmReadings.log"))
no_injection_df = dict_to_df(getData("noInjection.log"))

print("\n(1) Injection of FF as Speed\n")
print(fff_injection_df)

print("\n(2) Injection of RPM\n")
print(rpm_injection_df)

print("\n(3) No injection\n")
print(no_injection_df)


#Task 2 -- For creating plots
def create_plots(df):
    
    #Change of Speed over time
    plt.plot(df['Speed'])
    plt.title('Speed Over time')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.show()
    
    #Change of RPM over time
    plt.plot(df['RPM'])
    plt.title('RPM Over time')
    plt.xlabel('RPM')
    plt.ylabel('Speed')
    plt.show()
    
    #Relationship between Speed and RPM
    plt.scatter( df['Speed'],df['RPM'])
    plt.title('Speed vs RPM')
    plt.ylabel('RPM')
    plt.xlabel('Speed')
    plt.show()


# create_plots(fff_injection_df)
# create_plots(rpm_injection_df)
# create_plots(no_injection_df)


def freq_plot(df):

  #Frequency plot for speed
  plt.title('Speed Frequency')
  plt.hist(df['Speed'], bins=20)
  plt.xlabel('Speed')
  plt.ylabel('Frequency')
  plt.show()


  #Frequency plot for RPM
  plt.title('RPM Frequency')
  plt.hist('RPM')
  plt.xlabel('RPM')
  plt.ylabel('Frequency')
  plt.show()

  
# freq_plot(fff_injection_df)
# freq_plot(rpm_injection_df)
# freq_plot(no_injection_df)

def pearson_correlation(df):

  speed = list(df['Speed'])
  rpm = list(df['RPM'])

  # Apply the pearsonr()
  corr, p_value = pearsonr(speed, rpm)

  return corr, p_value

results = [
    ('Speed Injection', *pearson_correlation(fff_injection_df)),
    ('RPM Injection', *pearson_correlation(rpm_injection_df)),
    ('No Injection', *pearson_correlation(no_injection_df))
]

# correlation_table = pd.DataFrame(results, columns=['Files', 'Correlation', 'P_value'])

# print(correlation_table)



