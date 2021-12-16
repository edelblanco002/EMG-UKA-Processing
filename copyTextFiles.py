from globalVars import DIR_PATH, CORPUS
import os
from shutil import copyfile

#dirname: root folder
def main():
    
    for speaker in os.listdir(DIR_PATH + '/Transcripts'):
        for session in os.listdir(DIR_PATH + '/Transcripts/' + speaker):
            for file in os.listdir(DIR_PATH + '/Transcripts/' + speaker + '/' + session):
                srcFilename = DIR_PATH + '/Transcripts/' + speaker + '/' + session + '/' + file
                dstFilename = DIR_PATH + '/audioSync/' + speaker + '/' + session + '/' + file.replace('transcript','a')
                dstFilename = dstFilename.replace('.txt','.lab')
                copyfile(srcFilename,dstFilename)

    """if CORPUS == 'Pilot Study': # In Pilot Study, transcription files are named just with utterance number. They must be renamed
        for speaker in os.listdir(f"{DIR_PATH}/audioSync"):
            for session in os.listdir(f"{DIR_PATH}/audioSync/{speaker}"):
                for file in os.listdir(f"{DIR_PATH}/audioSync/{speaker}/{session}"):
                    if file.endswith('.wav'):
                        dstFilename = f"{DIR_PATH}/audioSync/{speaker}/{session}/{file.replace('.wav','.lab')}"
                        uttNumber = file[-8:-4]
                        if uttNumber != '0106': # 106 is the silent utterance
                            srcFilename = f"{DIR_PATH}/audioSync/{speaker}/{session}/{uttNumber}.lab"
                            copyfile(srcFilename,dstFilename)"""
if __name__ == "__main__":
    main()
