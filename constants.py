import os
import platform

if platform.system() == "Windows":
	user_root = "G:/"
	shared = user_root+"Drives compartilhados/"
	my_drive = user_root+"Meu Drive/"
else: 
	user_root = "/gdrive/"
	shared = user_root+"Shared drives/"
	my_drive = user_root+"My Drive/"


BASE_CONCORRENCIA = shared + "darwin/base-concorrencia/Base_Master/"
