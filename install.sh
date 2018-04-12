
#It will install it without support for CPU. If you use ti for training it will go quite slow
pip install keras==2.0.9 
pip install --upgrade tensorflow-gpu==1.4.0
pip install pattern stop-words numpy scipy matplotlib prettytable sklearn python-dateutil stop-words penman h5py

if ! [ -d "./resources" ]
	
then
	echo "Creating resources directory"
	mkdir resources
fi

#if ! [ -f "./resources/glove.6B.zip" ]
#then 
#	echo "Downloading external embeddings"
#	wget http://nlp.stanford.edu/data/glove.6B.zip -P ./resources/
#fi

#if ! [ -f "./resources/glove.6B.100d.txt" ] 
#then 
#	unzip resources/glove.6B.zip -d resources/
#fi

#Only used to then pick up some resources
if ! [ -d "./resources/jamr" ]
then 
	git clone https://github.com/jflanigan/jamr.git resources/jamr/
fi

if ! [ -d "./resources/amr-eager/" ]
then 
	git clone https://github.com/mdtux89/amr-eager resources/amr-eager/
fi 

if ! [ -f "./resources/amr-eager/resources_single.tar.gz" ]
then
	wget http://kinloch.inf.ed.ac.uk/public/direct/amreager/resources_single.tar.gz -P ./resources/amr-eager/
	tar xvzf ./resources/amr-eager/resources_single.tar.gz -C ./resources/amr-eager/
fi

if ! [-f "./resources/amr-naacl18-resources.zip" ]

then
	wget http://www.grupolys.org/software/amr-naacl18-resources.zip -P ./resources/
	unzip resources/amr-naacl18-resources.zip -d resources/
fi

if ! [-f "./resources/verbalization-list-v1.06.txt" ]
then
	wget https://amr.isi.edu/download/lists/verbalization-list-v1.06.txt -P ./resources/
fi

echo "Finished"


