import pandas as pd
import glob 

def make_site_list(input_csv): 
        """Get a list of all the snotel sites from input csv."""
        sites=pd.read_csv(input_csv) #read in the list of snotel sites by state
        try: 
            sites['site_num']=sites['site name'].str.extract(r"\((.*?)\)") #strip the site numbers from the column with the name and number
            site_ls= sites['site_num'].tolist()
            #print('try')
            #print(site_ls)
        except KeyError:  #this was coming from the space instead of _. Catch it and move on. 
            sites['site_num']=sites['site_name'].str.extract(r"\((.*?)\)") #strip the site numbers from the column with the name and number
            site_ls= sites['site_num'].tolist()
            print('except')

        try: 
            sites['huc_id']=sites['huc'].str.extract(r"\((.*?)\)")
            huc_dict=dict(zip(sites.site_num, sites.huc_id.str.slice(0,4,1))) #currently set to get huc04, change the end point to get a different huc level 
        except KeyError: 
            print('There was an issue with the format of the huc col, please double check inputs')
        
        return (sites,site_ls,huc_dict)
nwcc_file = pd.read_csv("/vol/v1/general_files/user_files/ben/excel_files/NWCC_high_resolution_coordinates_2019_edited.csv")

output_dict = {}
for file in glob.glob('/vol/v1/general_files/user_files/ben/excel_files/final_edited_files/*.csv'): 
    #state_dict = {}
    df = make_site_list(file)[0] #get the df from func above 
    state_dict = dict(zip(df['site_num'], df['huc_id']))
    output_dict.update(state_dict)

nwcc_file['huc_id'] = nwcc_file['site_num'].astype('str').map(output_dict)
print(nwcc_file.head())
#write out
nwcc_file.to_csv("/vol/v1/general_files/user_files/ben/excel_files/NWCC_high_resolution_coordinates_2019_hucs.csv",index=False)