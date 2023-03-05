 #include <stdlib.h>
#include <stdio.h>
#include <math.h>

//declarations

#define MIN(x,y)  ((x)<(y) ? (x) : (y))
#define MAX(x,y)  ((x)>(y) ? (x) : (y))
#define MIN_DOUBLE -HUGE_VAL
#define MAX_DOUBLE HUGE_VAL
#define LOW 0.1
#define HIGH .9
#define BIAS 1
#define sqr(x)   ((x)*(x))

//declare neural network structure

typedef struct {
int Units;  //number of neurons in the network
double * Outputs; //outputs of each neuron
double * Errors; //error terms
double ** Weights; //multi dimensional array of weights
double ** weightsSaved; //weights that we save
double ** deltaWeight; //weight adjustments
}Layer;

typedef struct {
Layer **       hiddenLayers; //array of hidden layers
Layer *        InputLayer; //input layer
Layer *        OutputLayer; //output layer
double         Alpha; //momentum factor
double         Eta; //Learning rate
double         Gain; //gain of the sigmoid function
double         Error; //total network error
}Network;

//now we initiate our random functions

void InitializeRandoms(){
srand(4711);
}

int intRandom(int low, int high){
return rand() % (high-low+1) + low;
}

double doubleRandom(double low, double high){
return ((double)rand()/RAND_MAX) * (high-low)+low;
}

//network application specific code


#define NUM_LAYERS 3    //could change number of layers
#define N 30            //could change N to change input layer
#define M 1              //could change M to change output layer
int Units[NUM_LAYERS]={N,30,M};  //here we are setting the number of neurons in each Layer
#define FIRST_YEAR 1700    //here we are defining the first year of of training data that the network will process
#define NUM_YEARS 280     //here we are defining the number of years that we will process in the neural network

#define TRAIN_LWB (N)    //THE STARTING YEAR OF SAMPLED TRAINING DATA
#define TRAIN_UPB (179)    //THE ENDING YEAR OF TRAINING DATA
#define TRAIN_YEARS (TRAIN_UPB-TRAIN_LWB+1)  //the number of years of data for the network to train on
#define TEST_LWB (180)  //STARTING YEAR FOR TESTING THE NETWORK
#define TEST_UPB (259)  //ENDING YEAR FOR TESTING THE NETWORK
#define TEST_YEARS (TEST_UPB-TEST_LWB+1) //number of years of data for training purposes
#define EVAL_LWB 260 //the year for network evaluation purposes
#define EVAL_UPB (NUM_YEARS-1) //the ending year of the evaluation process
#define EVAL_YEARS (EVAL_UPB-EVAL_LWB+1) //the number of years of data for the network to evaluate it's performance

//define array's of training data, YOU COULD CHANGE THIS DATA AND THE DATA ABOVE ^^ TO CHANGE INPUT DATA
double SunSpots_[NUM_YEARS];
double SunSpots  [NUM_YEARS]={0.0262,  0.0575,  0.0837,  0.1203,  0.1883,  0.3033,
                        0.1517,  0.1046,  0.0523,  0.0418,  0.0157,  0.0000,
                        0.0000,  0.0105,  0.0575,  0.1412,  0.2458,  0.3295,
                        0.3138,  0.2040,  0.1464,  0.1360,  0.1151,  0.0575,
                        0.1098,  0.2092,  0.4079,  0.6381,  0.5387,  0.3818,
                        0.2458,  0.1831,  0.0575,  0.0262,  0.0837,  0.1778,
                        0.3661,  0.4236,  0.5805,  0.5282,  0.3818,  0.2092,
                        0.1046,  0.0837,  0.0262,  0.0575,  0.1151,  0.2092,
                        0.3138,  0.4231,  0.4362,  0.2495,  0.2500,  0.1606,
                        0.0638,  0.0502,  0.0534,  0.1700,  0.2489,  0.2824,
                        0.3290,  0.4493,  0.3201,  0.2359,  0.1904,  0.1093,
                        0.0596,  0.1977,  0.3651,  0.5549,  0.5272,  0.4268,
                        0.3478,  0.1820,  0.1600,  0.0366,  0.1036,  0.4838,
                        0.8075,  0.6585,  0.4435,  0.3562,  0.2014,  0.1192,
                        0.0534,  0.1260,  0.4336,  0.6904,  0.6846,  0.6177,
                        0.4702,  0.3483,  0.3138,  0.2453,  0.2144,  0.1114,
                        0.0837,  0.0335,  0.0214,  0.0356,  0.0758,  0.1778,
                        0.2354,  0.2254,  0.2484,  0.2207,  0.1470,  0.0528,
                        0.0424,  0.0131,  0.0000,  0.0073,  0.0262,  0.0638,
                        0.0727,  0.1851,  0.2395,  0.2150,  0.1574,  0.1250,
                        0.0816,  0.0345,  0.0209,  0.0094,  0.0445,  0.0868,
                        0.1898,  0.2594,  0.3358,  0.3504,  0.3708,  0.2500,
                        0.1438,  0.0445,  0.0690,  0.2976,  0.6354,  0.7233,
                        0.5397,  0.4482,  0.3379,  0.1919,  0.1266,  0.0560,
                        0.0785,  0.2097,  0.3216,  0.5152,  0.6522,  0.5036,
                        0.3483,  0.3373,  0.2829,  0.2040,  0.1077,  0.0350,
                        0.0225,  0.1187,  0.2866,  0.4906,  0.5010,  0.4038,
                        0.3091,  0.2301,  0.2458,  0.1595,  0.0853,  0.0382,
                        0.1966,  0.3870,  0.7270,  0.5816,  0.5314,  0.3462,
                        0.2338,  0.0889,  0.0591,  0.0649,  0.0178,  0.0314,
                        0.1689,  0.2840,  0.3122,  0.3332,  0.3321,  0.2730,
                        0.1328,  0.0685,  0.0356,  0.0330,  0.0371,  0.1862,
                        0.3818,  0.4451,  0.4079,  0.3347,  0.2186,  0.1370,
                        0.1396,  0.0633,  0.0497,  0.0141,  0.0262,  0.1276,
                        0.2197,  0.3321,  0.2814,  0.3243,  0.2537,  0.2296,
                        0.0973,  0.0298,  0.0188,  0.0073,  0.0502,  0.2479,
                        0.2986,  0.5434,  0.4215,  0.3326,  0.1966,  0.1365,
                        0.0743,  0.0303,  0.0873,  0.2317,  0.3342,  0.3609,
                        0.4069,  0.3394,  0.1867,  0.1109,  0.0581,  0.0298,
                        0.0455,  0.1888,  0.4168,  0.5983,  0.5732,  0.4644,
                        0.3546,  0.2484,  0.1600,  0.0853,  0.0502,  0.1736,
                        0.4843,  0.7929,  0.7128,  0.7045,  0.4388,  0.3630,
                        0.1647,  0.0727,  0.0230,  0.1987,  0.7411,  0.9947,
                        0.9665,  0.8316,  0.5873,  0.2819,  0.1961,  0.1459,
                        0.0534,  0.0790,  0.2458,  0.4906,  0.5539,  0.5518,
                        0.5465,  0.3483,  0.3603,  0.1987,  0.1804,  0.0811,
                        0.0659,  0.1428,  0.4838,  0.8127 };
double Mean;   //average value
double TrainError; //the training error
double TrainErrorPredictingMean; //the average training error
double TestError; //the testing error of the network
double TestErrorPredictingMean; //the average error of network when in testing mode
//FILE * f;   if we wanted the output to be in a file

//this function normalizes our data array (to manipulate our data so that the numbers fit a predictable, uniform range)
void NormalizeData(){
int Year;
double Min,Max;

Min=MAX_DOUBLE;   //we set out minimum variable equal to the maximum double value
Max=MIN_DOUBLE;   //we set our maximum value to the minimum double value

for(Year=0;Year<NUM_YEARS;Year++){
  Min=MIN(Min,SunSpots[Year]); //take the smaller value of the maximum double value and the data input
  Max=MAX(Max,SunSpots[Year]); //take the larger value of the minimum double value and the data input
}
Mean=0;
//manipulate values so that they are fit to a smaller and more uniform range (E.G .1-1)
for(Year=0;Year<NUM_YEARS;Year++){
    SunSpots_[Year]=
    SunSpots[Year]=((SunSpots[Year]-Min)/(Max-Min))*(HIGH-LOW)+LOW;
    Mean+= SunSpots[Year]/NUM_YEARS;
}
}

//INITIALIZE APPLICATION
void InitializeApplication(Network * NET){
int Year, i;  //initialize loop variables
double Out,Err;
//initialize basic network variables
NET->Alpha=0.5;  //momentum
NET->Eta=0.05;   //learning rate
NET->Gain=1;     //sigmoid function

//Normalize data
NormalizeData();
TrainErrorPredictingMean=0;
//calculate testErrorPredictingMean
for(Year=TRAIN_LWB; Year<=TRAIN_UPB;Year++){
        //Loop through each unit in the output layer to calculate the Training Error when Predicting the Mean value of sunspots
    for(i=0;i<M;i++){
        Out=SunSpots[Year+i];
Err=Mean-Out;
TrainErrorPredictingMean+=0.5*sqr(Err);
    }
}

//now loop through each value and calculate the error when testing the network
for(Year=TEST_LWB; Year<=TEST_UPB;Year++){
    //loop through each output node
    for(i=0;i<M;i++){
        Out=SunSpots[Year+i];
        Err=Mean-Out;
        TestErrorPredictingMean+=0.5*sqr(Err);
    }
}


}
//Initialize Network

  void GenerateNetwork(Network *NET){
  int l,i;
  //ALLOCATE MEMORY FOR EACH LAYER AND NODE OF THE NETWORK
  NET->hiddenLayers= (Layer **) calloc(NUM_LAYERS,sizeof(Layer *));
//loop through each layer and allocate memory
for(l=0;l<NUM_LAYERS;l++){
    NET->hiddenLayers[l]=(Layer *)malloc(sizeof(Layer));

    NET->hiddenLayers[l]->Units = Units[l];  //set number of neurons in each layer
    NET->hiddenLayers[l]->Outputs=(double *) calloc(Units[l]+1,sizeof(double)); //allocate enough space in neuron output array to hold weights for next layer
NET ->hiddenLayers[l]->Errors=(double *) calloc (Units[l]+1,sizeof(double)); //allocate memory for error array
NET ->hiddenLayers[l]->Weights=(double **)calloc(Units[l]+1,sizeof(double *)); //allocate enough memory for holding the weights to calculate the input value of the next layer
NET->hiddenLayers[l]->weightsSaved =(double **)calloc (Units[l]+1,sizeof(double *)); //allocate memory for array of saved weights
NET->hiddenLayers[l]->deltaWeight=(double **)calloc (Units[l]+1,sizeof(double *)); //allocate memory for array of weight deltas
NET->hiddenLayers[l]->Outputs[0]=BIAS;

//continue allocating various weight arrays

if (l!=0){
for(i=1;i<=Units[l];i++){
        //allocate weights in this layer based on the number of units in the previous layer
    NET->hiddenLayers[l]->Weights[i]=(double *)calloc(Units[l-1]+1,sizeof(double));
    NET->hiddenLayers[l]->weightsSaved[i]=(double *)calloc(Units[l-1]+1,sizeof(double));
    NET->hiddenLayers[l]->deltaWeight[i]=(double *)  calloc(Units[l-1]+1,sizeof(double));

}
}
}
//Set Network Values
NET->InputLayer=NET->hiddenLayers[0];  //set input layer to first layer in layer array
NET->OutputLayer=NET->hiddenLayers[NUM_LAYERS-1]; //set output layer to last layer in layer array
NET->Alpha=0.9;     //set momentum factor
NET->Eta=0.25;      //set learning rate
NET->Gain=1;        //set gain




  }


  //function to generate random weights for the network
  void RandomWeights(Network * NET){
  //INITIALIZE VARIABLES
  int l,i,j;
  for(l=1;l<NUM_LAYERS;l++){
    for(i=1;i<=NET->hiddenLayers[l]->Units;i++){

        for(j=0;j<=NET->hiddenLayers[l-1]->Units;j++){
            //set random weights
            NET->hiddenLayers[l]->Weights[i][j]=doubleRandom(-0.5,0.5);

        }
    }
  }
  }

  //now set inputs
   void setInput(Network *NET,double *Inputs){
       int i;
       for(i=1;i<=NET->InputLayer->Units;i++){

        NET->InputLayer->Outputs [i]=Inputs[i-1];
       }

   }

   //get output
   void getOutput(Network *NET, double *Output){
       int i;
       for(i=1;i<=NET->OutputLayer->Units;i++){
        Output[i-1]=NET->OutputLayer->Outputs[i];
       }

   }

   //stopped training features

   void SaveWeights(Network * NET){
   int l,i,j;
   for(l=1;l<NUM_LAYERS;l++){
    for(i=1;i<=NET->hiddenLayers[l]->Units;i++){
        for(j=0;j<=NET->hiddenLayers[l-1]->Units;j++){
                NET->hiddenLayers[l]->weightsSaved[i][j]=NET->hiddenLayers[l]->Weights[i][j];

        }
    }
   }
   }

   //function to restore weights
   void RestoreWeights(Network *NET){
       int l,i,j;
       for(l=1;l<NUM_LAYERS;l++){
        for(i=1;i<=NET->hiddenLayers[l]->Units;i++){
            for(j=0;j<=NET->hiddenLayers[l-1]->Units;j++){
                NET->hiddenLayers[l]->Weights[i][j]=NET->hiddenLayers[l]->weightsSaved[i][j];
            }
        }
       }

    }

    //Functions to propagate network
    void propigateLayer(Network *NET,Layer *lower,Layer *upper){
        int i,j;
        double sum;
        for(i=1;i<=upper->Units;i++){
                sum=0;
            for(j=0;j<=lower->Units;j++){
                sum+=upper->Weights[i][j]*lower->Outputs[j];
            }
        upper->Outputs[i]=1.0/(1+exp(-NET->Gain*sum));
        }
    }

    //now propigate the entire network
    void PropigateNetwork(Network *NET){
    int l;
    for (l=0;l<NUM_LAYERS-1;l++){
        propigateLayer(NET,NET->hiddenLayers[l],NET->hiddenLayers[l+1]);
    }
    }


    //back propigation
void ComputeOutputError(Network *NET,double *target){
    int i;
    double Error,Out;
NET->Error=0;
    for(i=1;i<=NET->OutputLayer->Units;i++){
//FOR EACH OUTPUT NEURON CALCULATE THE ERROR AND ADD IT TO THE TOTAL NETWORK ERROR
        Out=NET->OutputLayer->Outputs[i];
        Error=target[i-1]-Out;
        NET->OutputLayer->Errors[i]=NET->Gain*Out*(1-Out)*Error;
        NET->Error+=0.5*sqr(Error);
    }

}

//back propigation
void BackPropigateLayer(Network *NET, Layer *Lower, Layer *Upper){
int i,j;
double Error,Out;
for(i=1;i<=Lower->Units;i++){
        Out=Lower->Outputs[i];  //get output of lower layer for each lower neuron
Error=0;
    for(j=1;j<=Upper->Units;j++){
            //get the summation of Error*Weight of the upper layer
       Error+= Upper->Weights[j][i]*Upper->Errors[j];
    }
    //multiply the error*weight summation by the derivitive of the Neuron Function to get the error array for the lower layer
    Lower->Errors[i]=NET->Gain*Out*(1-Out)*Error;

}
}

void BackPropigateNetwork(Network *NET){
    int l;
    for(l=NUM_LAYERS-1;l>1;l--){
        BackPropigateLayer(NET,NET->hiddenLayers[l-1],NET->hiddenLayers[l]);
    }
}

void AdjustWeights(Network *NET){
int l,i,j;
double Error,Out,dWeight;
for(l=1;l<NUM_LAYERS;l++){
    for(i=1;i<=NET->hiddenLayers[l]->Units;i++){
        for(j=0;j<=NET->hiddenLayers[l-1]->Units;j++){
            Out=NET->hiddenLayers[l-1]->Outputs[j];  //get Output of Jth neuron in lower layer
            Error=NET->hiddenLayers[l]->Errors[i];    //get Error of Ith neuron in higher layer
            dWeight=NET->hiddenLayers[l]->deltaWeight[i][j];
            //ADJUST WEIGHTS BY MULTIPLYING THE LEARNING RATE BY THE ERROR OF THE NEURON BY THE OUTPUT OF THE PREVIOUS NEURON  AND ADD THE DELTA WEIGHT
            NET->hiddenLayers[l]->Weights[i][j]+=NET->Eta*Error*Out+NET->Alpha*dWeight;
            //CALCULATE CHANGE IN WEIGHTS BY THE LEARNING RATE BY THE ERROR OF THE NEURON BY THE OUTPUT OF ONE OF THE PREVIOUS NEURONS
            NET->hiddenLayers[l]->deltaWeight[i][j]=NET->Eta*Error*Out;
        }
    }

}
}

//Simulating the Network

void SimulateNet(Network *NET, double *Input,double *Output, double *Target, int Training){
    setInput(NET,Input); //set each input layer neuron to our desired value
    PropigateNetwork(NET); //forward propigate the network
    getOutput(NET,Output); //Get output array from the output layer
    ComputeOutputError(NET,Target); //calculate the output error and create our error array
if(Training==1){
    //if we are training then we back propigate the network to calculate our weight adjustments
    BackPropigateNetwork(NET);
    //then we adjust the weights (add the partial derivitives*learning rate to our existing weights)
    AdjustWeights(NET);
}

}

void TrainNetwork(Network *NET,int trainNum){
int Year,n;
double Output[M];
for(n=0;n<TRAIN_YEARS*trainNum;n++){
   Year=intRandom(TRAIN_LWB,TRAIN_UPB); //generate random year
   SimulateNet(NET,&(SunSpots[Year-N]),Output,&(SunSpots[Year]),1); //simulate the network in training mode with 30 years of sunspots and the current sunspot as expected output
}
}

void TestNetwork(Network *NET){

    int Year;
    double Output[M];
    TrainError=0;
//Training Error
    for(Year=TRAIN_LWB;Year<=TRAIN_UPB;Year++){
            SimulateNet(NET,&(SunSpots[Year-N]),Output,&(SunSpots[Year]),0);
    TrainError+=NET->Error;

    }
//Testing Error
TestError=0;
for(Year=TEST_LWB;Year<=TEST_UPB;Year++){
    SimulateNet(NET,&(SunSpots[Year-N]),Output,&(SunSpots[Year]),0);
TestError+=NET->Error;
}
printf("Network Error is %0.3f on Training Set and %0.3f on Test Set\n",TestError/TestErrorPredictingMean,TrainError/TrainErrorPredictingMean);


}


//evaluate the network
void EvaluateNetwork(Network *NET){
    int Year;
    double Output[M];
    double Output_[M];

    printf("\n \n \n");
    printf("Year      SunSpots      Open Loop Prediction      Closed Loop Prediction\n");
    printf("\n");
    for(Year=EVAL_LWB;Year<=EVAL_UPB;Year++){
        SimulateNet(NET,&(SunSpots[Year-N]),Output,&(SunSpots[Year]),0);
        SimulateNet(NET,&(SunSpots_[Year-N]),Output_,&(SunSpots[Year]),0);
        SunSpots_[Year]=Output_[0];   //if output layer was more than one node then you would have to use a for loop here
        printf("%d      %0.3f               %0.3f                   %0.3f\n",FIRST_YEAR+Year,SunSpots[Year],Output[0],Output_[0]);   //if output layer was more than one node then you would have to use a for loop here
    }


}

//main function
void main(){
    //create Network
    Network NET;
    int Stop; //int BOOL
    double MinimumTestError=0;
    //initiate network
    InitializeRandoms();
    GenerateNetwork(&NET);
    RandomWeights(&NET);
    InitializeApplication(&NET);

    Stop=0;
    MinimumTestError=MAX_DOUBLE;

    do{

            //train the network
            TrainNetwork(&NET,1000);

            //test the network
            TestNetwork(&NET);
//IF OUT NETWORK IS MOVING IN THE RIGHT GRADIENT DIRECTION THEN WE SAVE THE WEIGHTS AND CONTINUE THE TRAINING
    if(TestError<MinimumTestError){
        printf("\n-saving weights....\n");
        MinimumTestError=TestError;
        SaveWeights(&NET);

    }

    //if we are no longer making progress with out graident descent then we stop the training and restore the previous weights
    else if(TestError>1.2*MinimumTestError){
        printf("\n-Stopping Training and restoring weights....\n");
        Stop=1; //stop the loop
        RestoreWeights(&NET);

    }

    }while(Stop==0);

//finish up and print results
TestNetwork (&NET);
EvaluateNetwork(&NET);
printf("Dean Urschel, 719-300-9671, urscheldean537@gmail.com\n");


}
