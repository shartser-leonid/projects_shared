#%%
import pandas as pd
import numpy as np
#%%
pd.set_option('display.max_rows', 40)
# %%
df=pd.DataFrame({'a':[1,2,3,4,5,6,7],'b':[5,6,7,8,9,10,11]})
# %%
df
# %%
df.loc[2:5]
# %%
# intialise data of lists.
data = {'Name':['Tom', 'nick', 'krish', 'jack'],
        'Age':[20, 21, 19, 18],'H':[1,2,3,4]}
 
# Create DataFrame
df = pd.DataFrame(data)
# %%
df
# %%
df.unstack(0)
# %%
data = pd.read_csv("nba.csv", index_col ="Name")
 
# retrieving row by loc method
first = data.loc["Avery Bradley"]
second = data.loc["R.J. Hunter"]
# %%
data
# %%
data.columns
# %%
dfm=pd.DataFrame(np.random.normal(size=[20,20]))
# %%
dfm.columns=[x for x in list('abcdefghijklmnopqrst')]
# %%
# %%
dfm['index']=[2*x for x in range(20)]
# %%
dfm=dfm.set_index('index')
# %%
dfm1=dfm.stack(0).unstack(0)
# %%
dfm1.index
# %%
dfm
# %%
dfri=dfm.unstack(0).reset_index()
# %%
dfri.columns=['k1','k2','val']
# %%
dfri.set_index(['k1','k2']).unstack(1)
# %%
data
# %%
sdf=data.loc[[xx for xx in data.index.dropna() if 'Ra' in xx.split(' ')[0]]]

# %%
# %%
sdf
# %%
sdf.sort_values("Number", axis = 1, ascending = True,
                 inplace = True, na_position ='last')
# %%
sdf
# %%
# importing pandas module
# Define a dictionary containing employee data
data1 = {'Name':['Jai', 'Anuj', 'Jai', 'Princi',
				'Gaurav', 'Anuj', 'Princi', 'Abhi'],
		'Age':[27, 24, 22, 32,
			33, 36, 27, 32],
        'Age2':[27, 24, 22, 32,
			33, 36, 27, 32],

		'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj',
				'Jaunpur', 'Kanpur', 'Allahabad', 'Aligarh'],
		'Qualification':['Msc', 'MA', 'MCA', 'Phd',
						'B.Tech', 'B.com', 'Msc', 'MA']}
	

# Convert the dictionary into DataFrame
df = pd.DataFrame(data1)

print(df)

# %%
df.groupby(axis=0,by=['Name','Qualification']).mean()
# %%

# example dataframe
example = {'Team':['Australia', 'England', 'South Africa',
				'Australia', 'England', 'India', 'India',
						'South Africa', 'England', 'India'],
						
		'Player':['Ricky Ponting', 'Joe Root', 'Hashim Amla',
					'David Warner', 'Jos Buttler', 'Virat Kohli',
					'Rohit Sharma', 'David Miller', 'Eoin Morgan',
												'Dinesh Karthik'],
												
		'Runs':[345, 336, 689, 490, 989, 672, 560, 455, 342, 376],
			
		'Salary':[34500, 33600, 68900, 49000, 98899,
					67562, 56760, 45675, 34542, 31176] }

df = pd.DataFrame(example)

total_salary = df['Salary'].groupby(df['Team'])

# printing the means value
print(total_salary.mean())	

# %%
total_salary.groups
# %%
df['Team']
# %%
df
# %%


# example dataframe
example = {'Team':['Arsenal', 'Manchester United', 'Arsenal',
				'Arsenal', 'Chelsea', 'Manchester United',
				'Manchester United', 'Chelsea', 'Chelsea', 'Chelsea'],
					
		'Player':['Ozil', 'Pogba', 'Lucas', 'Aubameyang',
					'Hazard', 'Mata', 'Lukaku', 'Morata',
										'Giroud', 'Kante'],
										
		'Goals':[5, 3, 6, 4, 9, 2, 0, 5, 2, 3] }

df = pd.DataFrame(example)

print(df)


total_goals = df['Goals'].groupby(df['Team'])
# %%
total_goals.sum()
# %%
df.groupby(by='Team',axis=0).sum()
# %%


x1=150
t1=1
t2=1/60*25
x_tgt = 200

x2=(x_tgt*(t1+t2) - x1*t1)/t2
x2
# %%
x2
# %%
data = {
            'A':[1, 2, 3],
            'B':[4, 5, 6],
            'C':[7, 8, 9] }
     
    # Convert the dictionary into DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)
def add(a,b,c):
    return a+b+c

df['add'] = df.apply(lambda row : add(row['A'],
                    row['B'], row['C']), axis = 1)



# %%
df
# %%

mulx = pd.MultiIndex.from_tuples([(15, 'Fifteen'), (19, 'Nineteen'),
(19, 'Fifteen'), (19, 'Nineteen')], names =['Num', 'Char'])
mulx.to_frame(index = True)


print(mulx.to_frame(index = True) )

# %%
mulx
# %%
pd.DataFrame().set_index(mulx)
# %%
d = {'num_legs': [4, 4, 2, 2],
     'num_wings': [0, 0, 2, 2],
     'class': ['mammal', 'mammal', 'mammal', 'bird'],
     'animal': ['cat', 'dog', 'bat', 'penguin'],
     'locomotion': ['walks', 'walks', 'flies', 'walks']}
df = pd.DataFrame(data=d)
df = df.set_index(['class', 'animal', 'locomotion'])
df

# %%
df.loc[(None,'cat',None)]
# %%
df.xs('walks',axis=0,level=2)
# %%
df
# %%
t=df.reset_index()
t[t['locomotion']=='walks']
# %%
