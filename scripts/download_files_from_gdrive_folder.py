# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 09:15:58 2017

@author: braatenj

https://googledrive.github.io/PyDrive/docs/build/html/index.html
https://pypi.python.org/pypi/PyDrive
"""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import sys
import time
import multiprocessing
from functools import partial


# define function to download tif files found in gDrive folder - called by multiprocessing.Pool().map()
def download_files(fileName, outDirPath):
  print('   file: '+fileName['title'])
  getFile = drive.CreateFile({'id': fileName['id']})
  getFile.GetContentFile(outDirPath+fileName['title']) # Download file

# get the arguments
#args = sys.argv
#gDirName = args[1]
#outDirPath = args[2]

# example of input9
gDirName = "daymet_outputs"
outDirPath = "/vol/v1/general_files/user_files/ben/excel_files/daymet_annual_data/daily_time_chunks/"
# make sure the paths end in '/'
if outDirPath[-1] != '/':
  outDirPath += '/'

os.chdir('/vol/v1/general_files/user_files/ben/python_files/sar_optical_low_snow_pnw/') #GoogleAuth looks in here for an authorization file - could pass the file as an argument and the get the os.path.dirname

# authenticate gDrive application and request access to gDrive account
gauth = GoogleAuth()
gauth.LocalWebserverAuth() # creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)     
                        
# find files in the specified gDrive folder
gDir = drive.ListFile({'q': "mimeType='application/vnd.google-apps.folder' and title contains '"+gDirName+"'"}).GetList()
print(gDir)
#if len(gDir) == 1: # TODO else print problem and exit
fileList = drive.ListFile({'q': "'"+gDir[0]['id']+"' in parents and title contains '.'"}).GetList()

# create the output folder if it does not already exist
if not os.path.isdir(outDirPath):
  os.mkdir(outDirPath)

# wait 10 seconds to start - if the folder is created in the line above
# then the download won't start, rerunning the script will get it to start
# could be that the folder is not fully registered before pool.map(func, fileList) 
# is called
time.sleep(10)

for i, thisFile in enumerate(fileList):
  print("i: "+str(i))
  download_files(thisFile, outDirPath)


# loop through downloading the files in parallel
#pool = multiprocessing.Pool(processes=3) 
#func = partial(download_files, outDirPath=outDirPath)
#pool.map(func, fileList)  
#pool.close()  
  
#!pip install -U -q PyDrive
# import os
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# #from google.colab import auth
# from oauth2client.client import GoogleCredentials

# # 1. Authenticate and create the PyDrive client.
# #auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)

# # choose a local (colab) directory to store the data.
# local_download_path = ''
# try:
#     os.makedirs("/vol/v1/general_files/user_files/ben/excel_files/daymet_annual_data/daily_time_chunks/")
# except: pass

# # 2. Auto-iterate using the query syntax
# #    https://developers.google.com/drive/v2/web/search-parameters
# file_list = drive.ListFile(
#     {'q': "'daymet_outputs' in parents"}).GetList()  #use your own folder ID here

# for f in file_list:
#     # 3. Create & download by id.
#     print('title: %s, id: %s' % (f['title'], f['id']))
#     fname = f['title']
#     print('downloading to {}'.format(fname))
#     f_ = drive.CreateFile({'id': f['id']})
#     f_.GetContentFile(fname)

# import the required libraries
# from __future__ import print_function
# import pickle
# import os.path
# import io
# import shutil
# import requests
# from mimetypes import MimeTypes
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# from __future__ import print_function
# import httplib2
# import os

# from apiclient import discovery
# from oauth2client import client
# from oauth2client import tools
# from oauth2client.file import Storage

# try:
#     import argparse
#     flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
# except ImportError:
#     flags = None

# # If modifying these scopes, delete your previously saved credentials
# # at ~/.credentials/drive-python-quickstart.json
# SCOPES = 'https://www.googleapis.com/auth/drive.metadata.readonly'
# CLIENT_SECRET_FILE = 'client_secret.json'
# APPLICATION_NAME = 'Drive API Python Quickstart'


# def get_credentials():
#     """Gets valid user credentials from storage.
#     If nothing has been stored, or if the stored credentials are invalid,
#     the OAuth2 flow is completed to obtain the new credentials.
#     Returns:
#         Credentials, the obtained credential.
#     """
#     home_dir = os.path.expanduser('~')
#     credential_dir = os.path.join(home_dir, '.credentials')
#     if not os.path.exists(credential_dir):
#         os.makedirs(credential_dir)
#     credential_path = os.path.join(credential_dir,
#                                    'drive-python-quickstart.json')

#     store = Storage(credential_path)
#     credentials = store.get()
#     if not credentials or credentials.invalid:
#         flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
#         flow.user_agent = APPLICATION_NAME
#         if flags:
#             credentials = tools.run_flow(flow, store, flags)
#         else: # Needed only for compatibility with Python 2.6
#             credentials = tools.run(flow, store)
#         print('Storing credentials to ' + credential_path)
#     return credentials

# def main():
#     """Shows basic usage of the Google Drive API.
#     Creates a Google Drive API service object and outputs the names and IDs
#     for up to 10 files.
#     """
#     credentials = get_credentials()
#     http = credentials.authorize(httplib2.Http())
#     service = discovery.build('drive', 'v3', http=http)

#     results = service.files().list(
#         pageSize=10,fields="nextPageToken, files(id, name)").execute()
#     items = results.get('files', [])
#     if not items:
#         print('No files found.')
#     else:
#         print('Files:')
#         for item in items:
#             print('{0} ({1})'.format(item['name'], item['id']))

# if __name__ == '__main__':
#     main()

# class DriveAPI:
#     global SCOPES
      
#     # Define the scopes
#     SCOPES = ['https://www.googleapis.com/auth/drive']
  
#     def __init__(self):
        
#         # Variable self.creds will
#         # store the user access token.
#         # If no valid token found
#         # we will create one.
#         self.creds = None
  
#         # The file token.pickle stores the
#         # user's access and refresh tokens. It is
#         # created automatically when the authorization
#         # flow completes for the first time.
  
#         # Check if file token.pickle exists
#         if os.path.exists('token.pickle'):
  
#             # Read the token from the file and
#             # store it in the variable self.creds
#             with open('token.pickle', 'rb') as token:
#                 self.creds = pickle.load(token)
  
#         # If no valid credentials are available,
#         # request the user to log in.
#         if not self.creds or not self.creds.valid:
  
#             # If token is expired, it will be refreshed,
#             # else, we will request a new one.
#             if self.creds and self.creds.expired and self.creds.refresh_token:
#                 self.creds.refresh(Request())
#             else:
#                 flow = InstalledAppFlow.from_client_secrets_file(
#                     'credentials.json', SCOPES)
#                 self.creds = flow.run_local_server(port=0)
  
#             # Save the access token in token.pickle
#             # file for future usage
#             with open('token.pickle', 'wb') as token:
#                 pickle.dump(self.creds, token)
  
#         # Connect to the API service
#         self.service = build('drive', 'v3', credentials=self.creds)
  
#         # request a list of first N files or
#         # folders with name and id from the API.
#         results = self.service.files().list(
#             pageSize=100, fields="files(id, name)").execute()
#         items = results.get('files', [])
  
#         # print a list of files
  
#         print("Here's a list of files: \n")
#         print(*items, sep="\n", end="\n\n")
  
#     def FileDownload(self, file_id, file_name):
#         request = self.service.files().get_media(fileId=file_id)
#         fh = io.BytesIO()
          
#         # Initialise a downloader object to download the file
#         downloader = MediaIoBaseDownload(fh, request, chunksize=204800)
#         done = False
  
#         try:
#             # Download the data in chunks
#             while not done:
#                 status, done = downloader.next_chunk()
  
#             fh.seek(0)
              
#             # Write the received data to the file
#             with open(file_name, 'wb') as f:
#                 shutil.copyfileobj(fh, f)
  
#             print("File Downloaded")
#             # Return True if file Downloaded successfully
#             return True
#         except:
            
#             # Return False if something went wrong
#             print("Something went wrong.")
#             return False
  
#     def FileUpload(self, filepath):
        
#         # Extract the file name out of the file path
#         name = filepath.split('/')[-1]
          
#         # Find the MimeType of the file
#         mimetype = MimeTypes().guess_type(name)[0]
          
#         # create file metadata
#         file_metadata = {'name': name}
  
#         try:
#             media = MediaFileUpload(filepath, mimetype=mimetype)
              
#             # Create a new file in the Drive storage
#             file = self.service.files().create(
#                 body=file_metadata, media_body=media, fields='id').execute()
              
#             print("File Uploaded.")
          
#         except:
              
#             # Raise UploadError if file is not uploaded.
#             raise UploadError("Can't Upload File.")
  
# if __name__ == "__main__":
#     obj = DriveAPI()
#     i = int(input("Enter your choice:1 - Download file, 2- Upload File, 3- Exit.\n"))
      
#     if i == 1:
#         f_id = input("Enter file id: ")
#         f_name = input("Enter file name: ")
#         obj.FileDownload(f_id, f_name)
          
#     elif i == 2:
#         f_path = input("Enter full file path: ")
#         obj.FileUpload(f_path)
      
#     else:
#         exit()