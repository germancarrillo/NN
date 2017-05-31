#include <iostream>
#include <iomanip>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

#define verbose 0

#define ZERO 0.0001

///////////////////////////////////////////////////////////////////////////////////////

class neuron{
  static uint totn;  
  uint id;
  //
  vector<double> inputs;
  vector<double> weights;
  vector<double> deltaw;
  vector<uint> connectionids;
  //
  double output;
  double graderror;
public:
  enum neurontype{in,hidden,out,bias}; 
  neurontype typen;  
  //
  neuron (neurontype,vector<neuron>); 
  double processvalue(vector<double> &inputs_,vector<neuron> &connectinglayer);
  double processerror(vector<neuron> &outputlayer,double &desiredvalue);  
  vector<double> updateweights(double &learningrate,double &momentum);
  //
  void setweight(uint i,double w){ weights[i]=w;};
  double getoutput(){return output;};
  double getgraderror(){return graderror;};
  double getweight(uint i){return weights[i];};
  uint getnumweights(){return weights.size();}
};

uint neuron::totn=0;

neuron::neuron(neurontype typen_,vector<neuron> nconnect){
  id=totn++;
  typen=typen_;
  graderror=(rand()%100/1000000.);
  for(auto i:nconnect){
    weights.push_back((rand()%100)/1000000.);
    deltaw.push_back(0);
    connectionids.push_back(i.id);	
  }
  if(typen==in){ weights.push_back(1); deltaw.push_back(0);}
  if(typen==bias) output=-1;

  if(verbose) cout<<"\t === INFO:: verbose:: neuron created, input size "<<inputs.size()<<" connectionsid size "<<connectionids.size()<<" weights size "<<weights.size()<<" neuron type "<<typen<<endl;
};

double neuron::processvalue(vector<double> &inputs_,vector<neuron> &connectinglayer){
  if(typen!=bias){
    inputs=inputs_; 
    output=0;  
    for(uint i=0;i<inputs.size();i++) output+= inputs[i]*weights[i];
    if(typen!=in) output = 1/(1+exp(-output));
  }
  if(verbose) cout<<"\t === INFO:: verbose:: neuron "<<id<<" output: "<<output<<endl;
  return output;
}; 

double neuron::processerror(vector<neuron> &outputlayer,double &desiredvalue){  
  if(typen==out) graderror = output*(1-output)*(desiredvalue-output);   
  else{
    double sumerroutput=0;
    for(auto n:outputlayer)
      for(uint i=0;i<n.connectionids.size();i++)
	if(n.connectionids[i]==id)
	  sumerroutput += n.weights[i]*n.graderror;
    graderror = output*(1-output)*sumerroutput;
  }
  if(verbose) cout<<"\t === INFO:: verbose:: neuron "<<id<<" gradient error: "<<graderror<<endl;
  return graderror;
}; 

vector<double> neuron::updateweights(double &learningrate,double &momentum){    
  for(uint i=0;i<weights.size();i++){
    deltaw[i]=learningrate*inputs[i]*graderror + momentum*deltaw[i];
    weights[i]+=deltaw[i];
  }
  if(verbose){ cout<<"\t === INFO:: verbose:: neuron "<<id<<" weights: "; for(auto w:weights) cout<<w<<", "; cout<<endl; }
  return weights; 
};

///////////////////////////////////////////////////////////////////////////////////////

class nnetwork{
  uint n_inputs;
  uint n_hidlayers;
  uint n_hidneurons;
  uint n_outneurons;
  uint n_epoch;
  double momentum;
  double learningrate;
  vector<double> desiredvalues;
  vector<vector<neuron>> layers;

  ofstream coutweights;
public:
  nnetwork (uint,uint,uint,uint);
  bool train(vector<vector<double>> &inputsample,uint n_epoch_,double learningrate_,double momentum_);
  bool estimatevalues(vector<double> event);
  bool estimateerrors();
  bool estimateweights();
  bool stoptraining(uint);
  bool test(vector<vector<double>> &inputsample);
  bool load(queue<double> &inputweights);
  void printoutput(){ cout<<"\t === INFO:: Output value(s): "; for(auto n:layers.back()) cout<<n.getoutput()<<", ";  cout<<endl; };
  void printerror(){ cout<<"\t === INFO:: Error value(s): "; for(auto n:layers.back()) cout<<n.getgraderror()<<", ";  cout<<endl; };
};

nnetwork::nnetwork(uint n_inputs_=0,uint n_hidlayers_=0,uint n_hidneurons_=0,uint n_outneurons_=0){
   
  n_inputs    =n_inputs_;
  n_hidlayers =n_hidlayers_;
  n_hidneurons=n_hidneurons_;
  n_outneurons=n_outneurons_;
  
  vector<neuron> empty;  
  cout<<"\t === INFO:: Creating NN with: "<<n_inputs<<" inputs, "<<n_hidlayers<<" hidden layer (at "<<n_hidneurons<<" neurons per hidden layer), and "<<n_outneurons<<" output"<<endl;
  
  vector<neuron> inlayer;                                                            // Input layer
  for(uint i=0;i<n_inputs;i++){ neuron a(neuron::in,empty); inlayer.push_back(a);}
  neuron biasin(neuron::bias,empty); inlayer.push_back(biasin); 
  layers.push_back(inlayer);
  cout<<"\t === INFO:: Input Layer created with "<<inlayer.size()-1<<" inputs, and 1 bias neuron, total of neurons: "<<inlayer.size()<<endl;
  
  for(uint l=0;l<n_hidlayers;l++){                                                   // Hidden layer(s)                          
    vector<neuron> hidlayer;
    for(uint n=0;n<n_hidneurons;n++){ neuron a(neuron::hidden,layers.back()); hidlayer.push_back(a); } 
    neuron biashid(neuron::bias,empty); hidlayer.push_back(biashid); 
    layers.push_back(hidlayer);
    cout<<"\t === INFO:: Hidden Layer created with "<<hidlayer.size()-1<<" core neurons, and 1 bias neuron, total of neurons: "<<hidlayer.size()<<endl;
  }
  
  vector<neuron> outlayer;                                                           // Output layer
  for(uint o=0;o<n_outneurons;o++){ neuron a(neuron::out,layers.back()); outlayer.push_back(a); desiredvalues.push_back(0);}
  layers.push_back(outlayer);
  cout<<"\t === INFO:: Output Layer created with "<<outlayer.size()<<" output neurons"<<endl;

  cout.setf(ios_base::fixed);  cout.precision(5);
};

bool nnetwork::estimatevalues(vector<double> event){
  vector<double> input,output;
  if(verbose) cout<<endl<<"\t === INFO:: verbose:: Calculating inputs"<<endl;    
  for(uint l=0;l<layers.size();l++){                                                        
    if(verbose) cout<<"\t === INFO:: verbose:: Inputs on layer "<<l<<endl;    
    input=output; output.clear();    
    vector<neuron> nextlayer; if(l<layers.size()-1) nextlayer=layers.at(l+1);    
    for(uint n=0;n<layers.at(l).size();n++){
      if(l==0 && layers.at(l).at(n).typen!=neuron::bias){ input.clear(); input.push_back(event.at(n)); }
      double outvalue=layers.at(l).at(n).processvalue(input,nextlayer);
      output.push_back(outvalue);
    }
  }
};

bool nnetwork::estimateerrors(){
  if(verbose) cout<<endl<<"\t === INFO:: verbose:: Estimating gradient errors"<<endl;
  for(int l=layers.size()-1;l>=0;l--)
    for(uint n=0;n<layers.at(l).size();n++){
      double desiredvalue=0; if(l==layers.size()-1) desiredvalue=desiredvalues[n];
      layers.at(l).at(n).processerror(layers.back(),desiredvalue);
    }
};

bool nnetwork::estimateweights(){
  if(verbose) cout<<endl<<"\t === INFO:: verbose:: Updating weights"<<endl;
  for(int l=layers.size()-1;l>=0;l--)
    for(uint n=0;n<layers.at(l).size();n++)
      layers.at(l).at(n).updateweights(learningrate,momentum);
  return true;
};

bool nnetwork::stoptraining(uint e){
  if(verbose) cout<<endl<<"\t === INFO:: verbose:: Checking stopping conditions"<<endl;
  double mse=0;
  for(uint n=0;n<layers.back().size();n++){
    double outputval=layers.back().at(n).getoutput();
    mse += pow(outputval - desiredvalues.at(n),2)/n_inputs;
    //if(outputval<ZERO*0.1 || outputval>1-ZERO*0.1){ cout<<"\t === INFO:: Stoping becasue output value found to be unreasonable "<<outputval<<endl; return true; }
  }
  if(mse<ZERO){    
    cout<<"\t === INFO:: Stoping after "<<e<<" epochs at a MSE of "<<mse<<endl; 
    coutweights.open("weights.dat"); coutweights.setf(ios_base::fixed);  coutweights.precision(12);    
    for(int l=0;l<layers.size();l++)
      for(uint n=0;n<layers.at(l).size();n++)
	for(uint i=0;i<layers.at(l).at(n).getnumweights();i++)
	  coutweights<<layers.at(l).at(n).getweight(i)<<endl;

    coutweights.close();
    return true;
  } 
  return false;
};


bool nnetwork::train(vector<vector<double>> &inputsample,uint n_epoch_=1,double learningrate_=0.5,double momentum_=0.5){  
  n_epoch     = n_epoch_;
  learningrate= learningrate_;
  momentum    = momentum_;
  
  uint totneurons=0; for(auto l:layers) for(auto n:l) totneurons++;
  cout<<"\t === INFO:: Training NN with: "<<inputsample.size()-1<<" input events. NN fromed by a total of "<<layers.size()<<" layers & "<<totneurons<<" neurons"<<endl;

  if(layers.size()<2) return false;
  uint eventid=0;
  for(auto event:inputsample){
    fill(desiredvalues.begin(),desiredvalues.end(),0);    
    uint desiredvalue=(int)event.front(); desiredvalues.at(desiredvalue)=1;
    event.erase(event.begin());    
    cout<<"\t === INFO:: Training, New event "<<eventid++<<", to desired value: "<<desiredvalue<<endl;
    for(uint e=0;e<n_epoch;e++){                
      this->estimatevalues(event);               
      this->estimateerrors();
      this->estimateweights();
      this->printoutput();
      if(this->stoptraining(e)) break;
    }
  }

  return true; 
}; 

bool nnetwork::test(vector<vector<double>> &inputsample){  
  cout<<"\t === INFO:: Testing NN on "<<inputsample.size()<<" input events"<<endl;
  for(auto event:inputsample){    
    if(verbose) cout<<endl<<"\t === INFO:: Testing, New event "<<endl;
    uint inputvalue=(int)event.front(); event.erase(event.begin());
    this->estimatevalues(event);
    double maxprob=0; int idval=-1;

    for(uint n=0;n<layers.back().size();n++) if(layers.back().at(n).getoutput()>maxprob){ maxprob=layers.back().at(n).getoutput(); idval=n; }
    cout<<endl<<"\t === INFO:: Testing event, written: "<<inputvalue<<" observed: "<<idval<<endl;
    this->printoutput(); 
  }  
  return true;  
}; 

bool nnetwork::load(queue<double> &inputweights){
  cout<<"\t === INFO:: Loading NN weights, total number of weights "<<inputweights.size()<<endl;
  
  for(int l=0;l<layers.size();l++)
    for(uint n=0;n<layers.at(l).size();n++)
      for(uint i=0;i<layers.at(l).at(n).getnumweights();i++){
	layers.at(l).at(n).setweight(i,inputweights.front());
	inputweights.pop();
      }

  return true;
};

///////////////////////////////////////////////////////////////////////////////////////

void loadinput(const char filename[],vector<vector<double>> &data){
  data.clear();
  ifstream f;  f.open(filename);     // Open File
  double var=0; string line;
  while(getline(f,line)){            // Read each line
    if(line.at(0)=='#') continue;
    vector<double> varset;    
    istringstream svar(line);        // Read each entry    
    while(svar >> var)
      varset.push_back(var);         // Fill vector
    data.push_back(varset);
  }
};

void loadweights(const char filename[],queue<double> &data){
  ifstream f;  f.open(filename);     
  double val=0;
  while(f >> val)
    data.push(val);  
};

///////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]){

  if(argc>3){
    char *train_filename    = argv[1];
    char *test_filename     = argv[2];
    char *validate_filename = argv[3];

    cout<<endl<<"\t === INFO:: New NN. Training sample "<<train_filename<<" testing "<<test_filename<<" validating "<<validate_filename<<endl;

    vector<vector<double>> inputsample_train;
    loadinput(train_filename,inputsample_train);
    
    vector<vector<double>> inputsample_test;
    loadinput(test_filename,inputsample_test);

    vector<vector<double>> inputsample_validate;
    loadinput(validate_filename,inputsample_test);
    
    //
    uint inputsize=inputsample_train.at(0).size()-1;
    uint hiddenlayers=1;
    uint neuronsinhiddenl=inputsize*2;
    uint outputs=10;
    //
    uint epochs=1000;
    double learningrate=0.0001;  
    double momentum=0.9;
    //
    nnetwork *NN = new nnetwork(inputsize,hiddenlayers,neuronsinhiddenl,outputs);  
    
    NN->train(inputsample_train,epochs,learningrate,momentum);
    NN->test(inputsample_test);    
  }else if(argc>2){
    char *weights_filename = argv[1];
    char *test_filename    = argv[2];

    cout<<endl<<"\t === INFO:: Loading NN. Weights from "<<weights_filename<<" testing input "<<test_filename<<endl;
    
    queue<double> inputweights;
    loadweights(weights_filename,inputweights);
    
    vector<vector<double>> inputsample_test;
    loadinput(test_filename,inputsample_test);
    
    //                                                                                                                                                                                                                                        
    uint inputsize=inputsample_test.at(0).size()-1;
    uint hiddenlayers=1;
    uint neuronsinhiddenl=inputsize*2;
    uint outputs=10;
    //                                                                                                                                                                                                                                         
    nnetwork *NN = new nnetwork(inputsize,hiddenlayers,neuronsinhiddenl,outputs);
    
    NN->load(inputweights);
    NN->test(inputsample_test);
    
  }else    
    cout<<"\t === INFO:: Arguments missing. Provide: train_filename test_filename and validation_filename, or weights_filename and test_filename"<<endl;
																			 
  return 0; 
}

 
