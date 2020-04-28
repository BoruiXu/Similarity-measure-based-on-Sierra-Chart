#include "sierrachart.h"
#include <string>
#include<cmath>
#include <omp.h>
using namespace std;
#define INF 99999
#define SUP -99999 
SCDLLName(" test studies _vali") 



//********************************************************************
// Sequence normalization function

float SMA(SCFloatArray  x,int length,int index){
	float sum = 0;
	float res = 0;
	for(int i=0;i<length;i++){
		sum+=x[index-i];
	}
	res = sum/length;
	return res;
}

float Standard_Deviation(SCFloatArray x,int length,int index){

	float sma = SMA(x,length,index);
	float sum = 0;
	float res ;
	float temp;
	for(int i=0;i<length;i++){
		temp = x[index-i]-sma;
		sum+=pow(temp,2);
	}

	sum=sum/length;
	res = sqrt(sum);
	return res;
}

void Z_Score(SCFloatArray s,int len,int index, float* res){
	float mean = SMA(s,len,index);
	float sd = Standard_Deviation(s,len,index);
	for(int i=0;i<len;i++){
		res[i] = (s[index-(len-1- i)]-mean)/sd;

	}
}

//********************************************************************
//*************************************************************
//dtw
float min_cost(float a,float b,float c){
	if(a<=b){
		if(a<=c){
			return a;
		}
		else{
			return c;
		}
	}
	else{
		if(b<=c){
			return b;
		}
		else{
			return c;
		}
	}
	return 0;
}



double DWT(SCFloatArray  x,SCFloatArray y,int len_x = 30,int len_y = 30,int index = 0){
	int len_x_new = len_x+1;
	int len_y_new = len_y+1;
	int i,j;

	float* temp_x;
	float* temp_y;
	temp_x = (float*)malloc( len_x*sizeof(float) );
	temp_y = (float*)malloc( len_y*sizeof(float) );
	Z_Score(x,len_x,index,temp_x);
	Z_Score(y,len_y,index,temp_y);



	float** temp;
	temp = (float**)malloc(sizeof(float*)*(len_x+1));
	for(i=0;i<=len_x;i++){
		temp[i] = (float*)malloc((len_y+1)*sizeof(float));
	}  

	

	for( i = 0;i<len_x_new;i++){
		for(j = 0;j<len_y_new;j++){
			temp[i][j] = INF;
		}
	}
	temp[0][0] = 0;

	for( i=0;i<len_x;i++){
		for( j=0;j<len_y;j++){
			float d = (temp_x[i]-temp_y[j]);
			float cost = d*d;
			temp[i+1][j+1] = min_cost(temp[i+1][j]+cost,temp[i][j+1]+cost,temp[i][j]+cost);
			
		}
	}

	double res = temp[len_x][len_y];
	free((void *)temp_x);
	free((void *)temp_y);
	for(i=0;i<=len_x;i++){
		 free((void *)temp[i]);
	}
	free((void *)temp);


	return (res);
}
//************************************************************
//parallel_dtw

float dtw_parallel(SCFloatArray s1, SCFloatArray s2, int len, int BarIndex, SCStudyInterfaceRef sc){

	int len_x_new = len+1;
	int len_y_new = len+1;

	float** cost_matrix;
	cost_matrix = (float**)malloc(sizeof(float*)*(len+1));
	for(int i=0;i<=len;i++){
		cost_matrix[i] = (float*)malloc((len+1)*sizeof(float));
	}
	float* temp_x;
	float* temp_y;
	temp_x = (float*)malloc( len*sizeof(float) );
	temp_y = (float*)malloc( len*sizeof(float) );

	Z_Score(s1,len,BarIndex,temp_x);
	Z_Score(s2,len,BarIndex,temp_y);

	// for(int i=0;i<len;i++){
	// 	temp_x[i] = s1[BarIndex-(len-1- i)];
	// 	temp_y[i] = s2[BarIndex-(len-1- i)];
	// }





	for(int i = 0;i<len_x_new;i++){
		for(int j = 0;j<len_y_new;j++){
			cost_matrix[i][j] = INF;
		}
	}
	cost_matrix[0][0] = 0;
	int loop_num = 2*len-1;

	
	for(int index=0;index<loop_num;index++){
		int calculation_num = 0;
		if(index<=len-1){
			calculation_num = index+1;
		}
		else{
			calculation_num = 2*len-index-1; 
		}


		

		#pragma omp parallel
		{
			int my_rank = omp_get_thread_num();
			int thread_count = omp_get_num_threads();
			// char alert[20];
			// sprintf_s(alert, "%d",thread_count);
			// sc.AddMessageToLog(SCString(alert),1);



			if(calculation_num<=thread_count){
				if(my_rank<=calculation_num-1){
					if(index<=len-1){
						float d = temp_x[my_rank] - temp_y[index-my_rank];
						float cost = d*d;
						cost_matrix[my_rank+1][index-my_rank+1] = cost+min_cost(cost_matrix[my_rank][index-my_rank+1],cost_matrix[my_rank+1][index-my_rank],cost_matrix[my_rank][index-my_rank]);					
					}
					else{
						int temp_num = index-len+1;
						float d = temp_x[temp_num+my_rank] - temp_y[len-my_rank-1];
						float cost = d*d;
						cost_matrix[temp_num+my_rank+1][len-my_rank] = cost+min_cost(cost_matrix[temp_num+my_rank][len-my_rank],cost_matrix[temp_num+my_rank+1][len-my_rank-1],cost_matrix[temp_num+my_rank][len-my_rank-1]);
					}
				}
				else{
					//do nothing
				}
			}
			else{
				int average_count = (int)(calculation_num/thread_count);
				
				float d=0;
				float cost =0;
				int start_num,end_num,temp_num;
				if(my_rank!=thread_count-1){
					start_num = my_rank*average_count+1;
					end_num = start_num+average_count-1;
					if(index<=len-1){
						for(int s = start_num;s<=end_num;s++){
							d = temp_x[s-1] - temp_y[index+1-s];
							cost = d*d;
							cost_matrix[s][index-s+2] = cost+min_cost(cost_matrix[s-1][index-s+2],cost_matrix[s][index-s+1],cost_matrix[s-1][index-s+1]);
						}
					}
					else{
						for(int s = start_num;s<=end_num;s++){
							temp_num = index-len;
							d = temp_x[temp_num+s]-temp_y[len-s];
							cost = d*d;
							cost_matrix[temp_num+s+1][len-s+1] = cost+min_cost(cost_matrix[temp_num+s][len-s+1],cost_matrix[temp_num+s+1][len-s],cost_matrix[temp_num+s][len-s]);
						}

					}
					
				}
				else{
					start_num = (thread_count-1)*average_count+1;
					end_num = calculation_num;

					if(index<=len-1){
						for(int s = start_num;s<=end_num;s++){
							d = temp_x[s-1] - temp_y[index+1-s];
							cost = d*d;
							cost_matrix[s][index-s+2] = cost+min_cost(cost_matrix[s-1][index-s+2],cost_matrix[s][index-s+1],cost_matrix[s-1][index-s+1]);
						}
					}
					else{
						for(int s = start_num;s<=end_num;s++){
							temp_num = index-len;
							d = temp_x[temp_num+s]-temp_y[len-s];
							cost = d*d;
							cost_matrix[temp_num+s+1][len-s+1] = cost+min_cost(cost_matrix[temp_num+s][len-s+1],cost_matrix[temp_num+s+1][len-s],cost_matrix[temp_num+s][len-s]);
						}
					}

				}
			}
		}



	}

	float res = cost_matrix[len][len]; 
	free((void *)temp_x);
	free((void *)temp_y);
	
	for(int i=0;i<=len;i++){
		free((void *)cost_matrix[i]);
	}
	free((void *)cost_matrix);
	return res;


}

 

//*************************************************************

//*************************************************************
//Euclidean_dis
float Euclidean_dis(SCFloatArray  x,SCFloatArray y,int len = 30,int index = 0){
	float res = 0;
	float* temp_x;
	float* temp_y;
	temp_x = (float*)malloc( len*sizeof(float) );
	temp_y = (float*)malloc( len*sizeof(float) );
	Z_Score(x,len,index,temp_x);
	Z_Score(y,len,index,temp_y);


	for(int i=0;i<len;i++){
		// float temp = x[index-i]-y[index-i];
		float temp = temp_x[i]- temp_y[i];
		float temp_dis =pow(temp,2);
		res+=temp_dis;
	}
	free((void *)temp_x);
	free((void *)temp_y);
	return sqrt(res);
}
//***************************************************************

//****************************************************************
//dtw using pattern recognition
int get_symbol_from_pattern(double p,int base){
	if(p>0){
		return base+2;
	}
	else if(p==0){
		return base+1;
	}
	else{
		return base;
	}
}


int get_pattern_dtw_symbol(SCFloatArray sequence, int len,int index){
	int res = 0;
	double p = sequence[len+index-1]-sequence[index];
	double min_value,max_value,average_value,bias,sum=0;
	min_value = INF;
	max_value = SUP;

	for(int i=index;i<len+index;i++){
		if(min_value>sequence[i]){
			min_value = sequence[i];
		}
		if(max_value<sequence[i]){
			max_value = sequence[i];
		}
		sum+=sequence[i];
	}

	average_value = sum/len;
	bias = (max_value - min_value)/3;

	if(min_value<=average_value && average_value<min_value+bias){
		res = get_symbol_from_pattern(p,1);
	}
	else if(min_value+bias<= average_value && average_value<= min_value+2*bias){
		res = get_symbol_from_pattern(p,4);
	}
	else{
		res = get_symbol_from_pattern(p,7);
	}

	return res;

}


float pattern_dtw(SCFloatArray xx,SCFloatArray yy,int len_x, int len_y, const int k,int index,SCStudyInterfaceRef sc){
	int len_x_new = (int)(len_x/k);
	int len_y_new = (int)(len_y/k);
	int* temp_x;
	int* temp_y;
	float** temp;
	temp_x = (int*)malloc( len_x_new*sizeof(int) );
	temp_y = (int*)malloc( len_y_new*sizeof(int) );
	temp = (float**)malloc(sizeof(float*)*(len_x_new+1));
	for(int i=0;i<=len_x_new;i++){
		temp[i] = (float*)malloc((len_y_new+1)*sizeof(float));
	}  


	
	// SCFloatArray temp_x = sim.Arrays[3];
	// SCFloatArray temp_y = sim.Arrays[4];

	int max_len = len_x_new;
	if(len_x_new<len_y_new){
		max_len = len_y_new;
	}
	
	for(int i=0;i<max_len;i++){
		if(i<len_x_new){
			temp_x[i] = get_pattern_dtw_symbol(xx,k,i*k+(index- len_x+1));
			if(temp_x[i]==0){
				sc.AddMessageToLog("error occured, a symbol is 0 in x_",1);
			}
		}
			if(i<len_y_new){
			temp_y[i] = get_pattern_dtw_symbol(yy,k,i*k+((index-len_y+1)));
			if(temp_y[i]==0){
				sc.AddMessageToLog("error occured, a symbol is 0 in y_ ",1);
			}
		}

	}
	

	//double temp[100][100];
	for(int i=0;i<len_x_new+1;i++){
		for(int j=0;j<len_y_new+1;j++){
			temp[i][j] = INF;
		}
	}

	temp[0][0]=0;
	for(int i=0;i<len_x_new;i++){
		for(int j=0;j<len_y_new;j++){
			float cost = 0;
			if(temp_x[i]!=temp_y[i]){
				cost = 1;
			}

			temp[i+1][j+1] = cost+min_cost(temp[i][j],temp[i+1][j],temp[i][j+1]);
		}
	}

	float res = (float)temp[len_x_new][len_y_new];
	
	free((void *)temp_x);
	free((void *)temp_y);
	for(int i=0;i<len_x_new;i++){
		free((void *)temp[i]);
	}
	free((void *)temp);
	return res;
}



//********************************************************************
//search query secquence
double LB_Yi(SCFloatArray query, SCFloatArray candidate, int window_size){
	double max_value = SUP;
	double min_value = INF;
	for(int i=0;i<window_size;i++){
		if(max_value<query[i]){
			max_value = query[i];
		}
		else if(min_value>query[i]){
			min_value = query[i];
		}
	}
	double distance=0;
	for(int i=0;i<window_size;i++){
		if(candidate[i]>max_value){
			distance+=pow((candidate[i]-max_value),2);
		}
		else if(candidate[i]<min_value){
			distance+=pow((candidate[i]-min_value),2);
		}
	}
	max_value = SUP;
	min_value = INF;
	double distance2 = 0;
	for(int i=0;i<window_size;i++){
		if(max_value<candidate[i]){
			max_value = candidate[i];
		}
		else if(min_value>candidate[i]){
			min_value = candidate[i];
		}
	}
	for(int i=0;i<window_size;i++){
		if(query[i]>max_value){
			distance2+=pow((query[i]-max_value),2);
		}
		else if(query[i]<min_value){
			distance2+=pow((query[i]-min_value),2);
		}
	}
	if(distance>distance2){
		return distance;
	}
	else{
		return distance2;
	}

}

int search_similarity(SCFloatArray query, SCFloatArray candidate, int window_size,int candidate_len,int thread_num,SCStudyInterfaceRef sc){
	float best_match = INF;
	int best_match_index = -1;
	
	sc.DataStartIndex = window_size-1;

	#pragma omp parallel for num_threads(thread_num)
	for(int BarIndex=sc.UpdateStartIndex;BarIndex<candidate_len;BarIndex++){
		

		int len_x_new = window_size+1;
		int len_y_new = window_size+1;
		int i,j;
		float res = INF;
		bool flag = true;

		float* temp_x;
		float* temp_y;
		temp_x = (float*)malloc( window_size*sizeof(float) );
		temp_y = (float*)malloc( window_size*sizeof(float) );
		Z_Score(query,window_size,window_size-1,temp_x);
		Z_Score(candidate,window_size,BarIndex,temp_y);


		float LB1 = pow((temp_x[window_size-1]-temp_y[window_size-1]),2); 
		float LB2 = pow((temp_x[0]-temp_y[0]),2);
		float LB = LB1+LB2;
		if(LB<best_match){

			float** temp;
			float temp_min ;
			temp = (float**)malloc(sizeof(float*)*(window_size+1));
			for(i=0;i<=window_size;i++){
				temp[i] = (float*)malloc((window_size+1)*sizeof(float));
			} 

			for( i = 0;i<len_x_new;i++){
				for(j = 0;j<len_y_new;j++){
					temp[i][j] = INF;
				}
			}
			temp[0][0] = 0;

			for( i=0;i<window_size;i++){
				temp_min = INF;
				for( j=0;j<window_size;j++){
					float d = (temp_x[i]-temp_y[j]);
					float cost = d*d;
					temp[i+1][j+1] = min_cost(temp[i+1][j]+cost,temp[i][j+1]+cost,temp[i][j]+cost);
					
					if(temp[i+1][j+1]<temp_min){
						temp_min = temp[i+1][j+1];
					}
					
				}
				if(temp_min>best_match){
					flag = false;
					break;
				}
			}
			res = temp[window_size][window_size];
			for(i=0;i<=window_size;i++){
				free((void *)temp[i]);
			}
			free((void *)temp);
		}
		else{
			flag = false;
		}
		if(flag){
			#pragma omp critical
			{
				if(res<best_match){
					best_match = res;
					best_match_index = BarIndex;
				}

			}
		}
		free((void *)temp_x);
		free((void *)temp_y);
		
	}
	return best_match_index;

}


int search_similarity_single_noLB(SCFloatArray query, SCFloatArray candidate, int window_size,int candidate_len,SCStudyInterfaceRef sc){
	float best_match = INF;
	int best_match_index = -1;
	
	sc.DataStartIndex = window_size-1;

	for(int BarIndex=sc.UpdateStartIndex;BarIndex<candidate_len;BarIndex++){
		

		int len_x_new = window_size+1;
		int len_y_new = window_size+1;
		int i,j;
		float res = INF;

		float* temp_x;
		float* temp_y;
		temp_x = (float*)malloc( window_size*sizeof(float) );
		temp_y = (float*)malloc( window_size*sizeof(float) );
		Z_Score(query,window_size,window_size-1,temp_x);
		Z_Score(candidate,window_size,BarIndex,temp_y);

		if(1){

			float** temp;
			temp = (float**)malloc(sizeof(float*)*(window_size+1));
			for(i=0;i<=window_size;i++){
				temp[i] = (float*)malloc((window_size+1)*sizeof(float));
			} 

			for( i = 0;i<len_x_new;i++){
				for(j = 0;j<len_y_new;j++){
					temp[i][j] = INF;
				}
			}
			temp[0][0] = 0;

			for( i=0;i<window_size;i++){
				for( j=0;j<window_size;j++){
					float d = (temp_x[i]-temp_y[j]);
					float cost = d*d;
					temp[i+1][j+1] = min_cost(temp[i+1][j]+cost,temp[i][j+1]+cost,temp[i][j]+cost);
					
					
				}
			}
			res = temp[window_size][window_size];
			for(i=0;i<=window_size;i++){
				free((void *)temp[i]);
			}
			free((void *)temp);
		}
		
			
		if(res<best_match){
			best_match = res;
			best_match_index = BarIndex;
		}

			
		
		free((void *)temp_x);
		free((void *)temp_y);
		
	}
	return best_match_index;

}

//********************************************************************

//get data from another symbol

int* GetSymbolData(int* ChartNumber ,const char* symbol1, int timeframe,bool historiy,bool second_symbol,const char* symbol2,int priorNumber,SCStudyInterfaceRef sc)
{
	ChartNumber[0] = 0;
	ChartNumber[1] = 0;
	if(historiy){
		s_ACSOpenChartParameters OpenChartParameters;
		//OpenChartParameters.Reset();
		OpenChartParameters.PriorChartNumber = priorNumber;
		OpenChartParameters.ChartDataType = DAILY_DATA;
		OpenChartParameters.HistoricalChartBarPeriod = HISTORICAL_CHART_PERIOD_DAYS;
		OpenChartParameters.Symbol = symbol1; 
		OpenChartParameters.DaysToLoad = 0;

		OpenChartParameters.HideNewChart = 1;

		ChartNumber[0] = sc.OpenChartOrGetChartReference(OpenChartParameters);

		if(second_symbol){
			s_ACSOpenChartParameters OpenChartParameters2;
		//OpenChartParameters.Reset();
			OpenChartParameters2.PriorChartNumber = ChartNumber[0];
			OpenChartParameters2.ChartDataType = DAILY_DATA;
			OpenChartParameters2.HistoricalChartBarPeriod = HISTORICAL_CHART_PERIOD_DAYS;
			OpenChartParameters2.Symbol = symbol2; 
			OpenChartParameters2.DaysToLoad = 0;

			OpenChartParameters2.HideNewChart = 1;

			ChartNumber[1] = sc.OpenChartOrGetChartReference(OpenChartParameters2);

		}
	}
	else{
		//sc.AddMessageToLog("test IsFullRecalculation",1);
		s_ACSOpenChartParameters OpenChartParameters;
		OpenChartParameters.PriorChartNumber = priorNumber;
		OpenChartParameters.ChartDataType = INTRADAY_DATA; //This can also be set to: DAILY_DATA
		OpenChartParameters.Symbol = symbol1;//When want to use the symbol of the chart the study function is on, use sc.GetRealTimeSymbol()
		//sc.AddMessageToLog(OpenChartParameters.Symbol,1);
		
		OpenChartParameters.IntradayBarPeriodType = IBPT_DAYS_MINS_SECS;
		OpenChartParameters.IntradayBarPeriodLength = timeframe*SECONDS_PER_MINUTE;  // 30 minutes
		OpenChartParameters.DaysToLoad = 0;//same as calling chart
		// These are optional
		//OpenChartParameters.SessionStartTime.SetTimeHMS(12, 0, 0);
		//OpenChartParameters.SessionEndTime.SetTimeHMS(23,59,59);
		//OpenChartParameters.EveningSessionStartTime.SetTimeHMS(0,0,0);
		//OpenChartParameters.EveningSessionEndTime.SetTimeHMS(23,59,59);
		OpenChartParameters.LoadWeekendData = 1;

		OpenChartParameters.UseEveningSession = 0;
		OpenChartParameters.HideNewChart = 1;

		ChartNumber[0] = sc.OpenChartOrGetChartReference(OpenChartParameters);
		// SCString DateText;
		// DateText.Format("%d", ChartNumber);
		// sc.AddMessageToLog(DateText,1);


		if(second_symbol){
			s_ACSOpenChartParameters OpenChartParameters2;
			OpenChartParameters2.PriorChartNumber = ChartNumber[0];
			OpenChartParameters2.ChartDataType = INTRADAY_DATA; //This can also be set to: DAILY_DATA
			OpenChartParameters2.Symbol = symbol2;//When want to use the symbol of the chart the study function is on, use sc.GetRealTimeSymbol()
			//sc.AddMessageToLog(OpenChartParameters.Symbol,1);
			
			OpenChartParameters2.IntradayBarPeriodType = IBPT_DAYS_MINS_SECS;
			OpenChartParameters2.IntradayBarPeriodLength = timeframe*SECONDS_PER_MINUTE;  // 30 minutes
			OpenChartParameters2.DaysToLoad = 0;//same as calling chart
			// These are optional
			//OpenChartParameters2.SessionStartTime.SetTimeHMS(12, 0, 0);
			//OpenChartParameters2.SessionEndTime.SetTimeHMS(23,59,59);
			//OpenChartParameters2.EveningSessionStartTime.SetTimeHMS(0,0,0);
			//OpenChartParameters2.EveningSessionEndTime.SetTimeHMS(23,59,59);
			OpenChartParameters2.LoadWeekendData = 1;

			OpenChartParameters2.UseEveningSession = 0;
			OpenChartParameters2.HideNewChart = 1;

			ChartNumber[1] = sc.OpenChartOrGetChartReference(OpenChartParameters2);
			// SCString DateText;
			// DateText.Format("%d", ChartNumber);
			// sc.AddMessageToLog(DateText,1);

		}
	}

	return ChartNumber;
}

//*******************************************************************

SCSFExport scsf_average_vali(SCStudyGraphRef sc){
	if (sc.SetDefaults)
	{
		
		sc.GraphName = "average_vali";
		
		sc.AutoLoop = 0;  // Automatic looping is enabled.

		sc.GraphRegion = 0;
		
		// During development set this flag to 1, so the DLL can be rebuilt without restarting Sierra Chart. When development is completed, set it to 0 to improve performance.
		sc.FreeDLL = 1; 
		
		sc.Subgraph[0].Name = "average_test_vali";
		sc.Subgraph[0].PrimaryColor = RGB(255,0,0);  // Red
		sc.Subgraph[0].DrawStyle = DRAWSTYLE_LINE;  // Look in scconstants.h for other draw styles
		
		
		return;
	}

	double sum = 0;
	for(int Index = sc.UpdateStartIndex;Index<sc.ArraySize;++Index){
		sum+=sc.BaseData[SC_LAST][Index];
	}
	sc.DataStartIndex = 0;
	float res = (float)(sum/sc.ArraySize);
	for (int Index = sc.UpdateStartIndex; Index < sc.ArraySize; ++Index)
	{	
		
    // fill in the first subgraph with the last values
    	sc.Subgraph[0][Index] = res;
	} 
}


SCSFExport scsf_distance(SCStudyGraphRef sc)
{
	// Section 1 - Set the configuration variables and defaults
	
	
	if (sc.SetDefaults)  
	{
		sc.GraphName = "distance";
		
		sc.StudyDescription = "Example function for calculating a simple moving average from scratch.";
		
		sc.AutoLoop = 0;  // fasle
		sc.FreeDLL = 1;
		
		// Set the Chart Region to draw the graph in.  Region 0 is the main
		// price graph region.
		sc.GraphRegion = 0;
		
		// Set the name of the first subgraph
		sc.Subgraph[0].Name = "distance";

		// Set the color and style of the subgraph line.  
		sc.Subgraph[0].PrimaryColor = RGB(0,0,255);  // Red
		sc.Subgraph[0].DrawStyle = DRAWSTYLE_LINE;
		
		
		
		// Must return before doing any data processing if sc.SetDefaults is set
		return;
	}
	
	
	
	for (int Index = sc.UpdateStartIndex; Index < sc.ArraySize; ++Index)
	{
    // fill in the first subgraph with the last values
		if(Index<sc.ArraySize-1){
			sc.Subgraph[0][Index] = (sc.BaseData[SC_LAST][Index]+sc.BaseData[SC_LAST][Index+1])/2;
		}
		else{
			sc.Subgraph[0][Index] = sc.BaseData[SC_LAST][Index];
		}

    
	} 
	
}


SCSFExport scsf_ReferenceStudyData(SCStudyInterfaceRef sc)
{
    SCSubgraphRef Average = sc.Subgraph[0];
    SCInputRef Study1 = sc.Input[0];
    SCInputRef Study1Subgraph = sc.Input[1];



    if (sc.SetDefaults)
    {
        sc.GraphName = "Reference Study Data";
        sc.StudyDescription = "This study function is an example of calculating the similarity.";		
        sc.AutoLoop = 1;

        // We must use a low precedence level to ensure
        // the other studies are already calculated first.
        sc.CalculationPrecedence = LOW_PREC_LEVEL;


        Average.Name = "Average";
        Average.DrawStyle = DRAWSTYLE_LINE;
        Study1.Name = "Input Study 1";
        Study1.SetStudyID(0);
        Study1Subgraph.Name = "Study 1 Subgraph";
        Study1Subgraph.SetSubgraphIndex(0);

        return;
    }

    // Get the Subgraph specified with the Study 1
    // Subgraph input from the study specified with
    // the Input Study 1 input.
    SCFloatArray Study1Array;

    sc.GetStudyArrayUsingID(Study1.GetStudyID(),Study1Subgraph.GetSubgraphIndex(),Study1Array);

    // We are getting the value of the study Subgraph
    // at sc.Index. For example, this could be
    // a moving average value if the study we got in
    // the prior step is a moving average.
    //float RefStudyCurrentValue = Study1Array[sc.Index];
    Average[sc.Index] = Study1Array[sc.Index]; 

    // Here we will add 10 to this value and compute 
    // an average of it. Since the moving average
    // function we will be calling requires an input
    // array, we will use one of the internal arrays
    // on a Subgraph to hold this intermediate 
    // calculation. This internal array could be 
    // thought of as a Spreadsheet column where you 
    // are performing intermediate calculations.

    // sc.Subgraph[0].Arrays[9][sc.Index] = RefStudyCurrentValue + 10;

    // sc.SimpleMovAvg(sc.Subgraph[0].Arrays[9],sc.Subgraph[0],15);
}
//****************************************************************************

SCSFExport scsf_similarity_pattern_dtw(SCStudyInterfaceRef sc){
	SCSubgraphRef sim = sc.Subgraph[0];
	SCSubgraphRef sim2 = sc.Subgraph[1];
	SCInputRef SYMBOL1 = sc.Input[9];
    SCInputRef SYMBOL1Subgraph = sc.Input[10];
    SCInputRef SYMBOL2 = sc.Input[11];
    SCInputRef SYMBOL2Subgraph = sc.Input[12];

	if(sc.SetDefaults){
		sc.GraphName = "similarity_pattern_dtw";

		sc.StudyDescription = "This function is an implementation of the modified dtw function using pattern recognition";

		sc.AutoLoop =0;
		sc.FreeDLL = 1;
		sc.CalculationPrecedence = LOW_PREC_LEVEL;

		sc.GraphRegion = 1;
		sc.Input[0].Name = "SYMBOL";
		sc.Input[0].SetString(sc.GetRealTimeSymbol());
		sc.Input[1].Name = "num_threads";
		sc.Input[1].SetInt(8);
		sc.Input[2].Name = "similarity window size ";
		sc.Input[2].SetInt(300);
		sc.Input[3].Name="pattern length";
		sc.Input[3].SetInt(10);
		sc.Input[4].Name="SYMBOL2";
		sc.Input[4].SetString(sc.GetRealTimeSymbol());
		sc.Input[5].Name="Second Symbol";
		sc.Input[5].SetYesNo(1);
		sc.Input[6].Name="historical data";
		sc.Input[6].SetYesNo(1);
		sc.Input[7].Name="SECONDS_PER_MINUTE";
		sc.Input[7].SetInt(1);
		sc.Input[8].Name = "Interpolation";
		sc.Input[8].SetYesNo(1);
		SYMBOL1.Name = "SYMBOL1_Interpolation";
        SYMBOL1.SetStudyID(0);
        SYMBOL1Subgraph.Name = "Price type for symbol 1";
        SYMBOL1Subgraph.SetSubgraphIndex(3);

        SYMBOL2.Name = "SYMBOL2_Interpolation";
        SYMBOL2.SetStudyID(0);
        SYMBOL2Subgraph.Name = "Price type for symbol 2";
        SYMBOL2Subgraph.SetSubgraphIndex(3);

		sim2.Name = "the result of similarity2";
		sim2.PrimaryColor = RGB(0,255,0);
		sim2.DrawStyle =DRAWSTYLE_LINE;

		sim.Name = "the result of similarity";
		sim.PrimaryColor = RGB(255,0,0);  // Red
		sim.DrawStyle = DRAWSTYLE_LINE;

		return;


	}

	int& ChartNumber = sc.GetPersistentInt(1);
	int& ChartNumber2 = sc.GetPersistentInt(2);
	int threads_count =  sc.Input[1].GetInt();
	int window_size = sc.Input[2].GetInt();
	int pattern_len = sc.Input[3].GetInt();
	bool second_symbol = sc.Input[5].GetYesNo();
	bool  historiy = sc.Input[6].GetYesNo();
	int minutes = sc.Input[7].GetInt();
	bool Interpolation = sc.Input[8].GetYesNo();

	if(threads_count<1){
		threads_count = 1;
		sc.Input[1].SetInt(1);

	}
	else if(threads_count>20){
		threads_count = 20;
		sc.Input[1].SetInt(20);
	}

	if(window_size<15){
		window_size=15;
	}
	else if(window_size>500){
		window_size=500;
	}

	if(pattern_len<1){
		pattern_len = 2;
		if(window_size% pattern_len!=0){
			int temp = (int)(window_size/pattern_len);
			for(int i=temp;i>5;i--){
				if(window_size%i==0){
					pattern_len = window_size/i;
					break;
				}
			}
			
			if(window_size% pattern_len!=0){
				sc.AddMessageToLog("the window_size is unsuitable!",1);
				return;
			}
			
		}
		sc.Input[3].SetInt(pattern_len);
	}
	else if(pattern_len >50){
		pattern_len = 50;

		if(window_size% pattern_len!=0){
			int temp = (int)(window_size/pattern_len);
			int  max_value = (int)(window_size/2);
			for(int i=temp;i<max_value;i++){
				if(window_size%i==0){
					pattern_len = window_size/i;
					break;
				}
			}
			if(window_size% pattern_len!=0){
				sc.AddMessageToLog("the window_size is unsuitable!",1);
				return;

			}
		}
		sc.Input[3].SetInt(pattern_len);
		
	}
	else if(window_size% pattern_len!=0){

		int temp = (int)(window_size/pattern_len);
		for(int i=temp;i>5;i--){
			if(window_size%i==0){
				pattern_len = window_size/i;
				break;
			}
		}
		if(window_size% pattern_len!=0){
			int  max_value = (int)(window_size/2);
			for(int i=temp;i<max_value;i++){
				if(window_size%i==0){
					pattern_len = window_size/i;
					break;
				}
			}
			if(window_size% pattern_len!=0){
				sc.AddMessageToLog("the window_size is unsuitable!",1);
				return;

			}
		}
		sc.Input[3].SetInt(pattern_len);
		
	}

	int ChartNumberArray[2];

	if (sc.IsFullRecalculation && !Interpolation)
	{
		
		//string symbol1, timeframe time,bool historiy,bool second_symbol,string symbol2,int priorNumber)
		GetSymbolData(ChartNumberArray ,sc.Input[0].GetString(),minutes,historiy,second_symbol,sc.Input[4].GetString(),ChartNumber,sc);
		ChartNumber = ChartNumberArray[0];
		ChartNumber2 = ChartNumberArray[1];
		
	}


	if ((ChartNumber != 0  && ChartNumber2!=0) ||(ChartNumber != 0  &&!second_symbol)|| Interpolation)
	{
		
		SCFloatArray S1Array,S2Array;

		SCGraphData ReferenceChartData;
		SCGraphData ReferenceChartData2;

		// Get the arrays from the reference chart
		if(!Interpolation){
			sc.GetChartBaseData(ChartNumber, ReferenceChartData);
			S1Array = ReferenceChartData[SC_LAST];
			if(second_symbol){
				sc.GetChartBaseData(ChartNumber2, ReferenceChartData2);
				S2Array = ReferenceChartData2[SC_LAST];
			}
		}
		else{
			sc.GetStudyArrayUsingID(SYMBOL1.GetStudyID(),SYMBOL1Subgraph.GetSubgraphIndex(),S1Array);

			if(second_symbol){
				sc.GetStudyArrayUsingID(SYMBOL2.GetStudyID(),SYMBOL2Subgraph.GetSubgraphIndex(),S2Array);
			}
		}
		
		
		if((S1Array.GetArraySize()>0 &&S2Array.GetArraySize()>0)||(S1Array.GetArraySize()>0 && !second_symbol)){
			sc.DataStartIndex = 0;
			#pragma omp parallel for num_threads(threads_count)
			for (int BarIndex = sc.UpdateStartIndex; BarIndex < sc.ArraySize; BarIndex++){
				
				//data1[BarIndex] = sc.BaseData[SC_LAST][BarIndex];
				//data2[BarIndex] = ReferenceChartData[SC_LAST][BarIndex];
				if(BarIndex>=window_size-1){
					 float res = pattern_dtw(sc.BaseData[SC_LAST],S1Array,window_size,window_size,pattern_len,BarIndex,sc);
					sim[BarIndex] =  res;
					if(second_symbol){
						float res2 = pattern_dtw(sc.BaseData[SC_LAST],S2Array,window_size,window_size,pattern_len,BarIndex,sc);
						sim2[BarIndex] = res2;
					}

				}
				
			}	
		}
		else{
			sc.AddMessageToLog("wrong!",1);
		}

	}
	else{
		sc.AddMessageToLog("open data wrong!",1);
	}

}

SCSFExport scsf_similarity_dtw_openmp(SCStudyInterfaceRef sc){
	SCSubgraphRef sim = sc.Subgraph[0];
	SCSubgraphRef sim2 = sc.Subgraph[1];
	SCInputRef SYMBOL1 = sc.Input[11];
    SCInputRef SYMBOL1Subgraph = sc.Input[12];
    SCInputRef SYMBOL2 = sc.Input[13];
    SCInputRef SYMBOL2Subgraph = sc.Input[14];

    if (sc.SetDefaults)  
	{
		sc.GraphName = "similarity_dtw_openmp";
		
		sc.StudyDescription = "This function is an implementation of the dtw function using openmp ";
		
		sc.AutoLoop = 0;  // true
		sc.FreeDLL = 1;
		sc.CalculationPrecedence = LOW_PREC_LEVEL;
		
		// Set the Chart Region to draw the graph in.  Region 0 is the main
		// price graph region.
		sc.GraphRegion = 1;

		sc.Input[0].Name = "SYMBOL";
		sc.Input[0].SetString(sc.GetRealTimeSymbol());
		sc.Input[1].Name = "num_threads";
		sc.Input[1].SetInt(8);
		// sc.Input[2].Name="Sequence normalization";
		// sc.Input[2].SetYesNo(1);
		// sc.Input[3].Name = "mean length for sequence normalization ";
		// sc.Input[3].SetInt(10);
		// sc.Input[4].Name = "standard_deviation length for sequence normalization";
		// sc.Input[4].SetInt(10);
		sc.Input[5].Name = "similarity window size ";
		sc.Input[5].SetInt(300);
		sc.Input[6].Name = "SYMBOL2";
		sc.Input[6].SetString(sc.GetRealTimeSymbol());
		sc.Input[7].Name="Second Symbol";
		sc.Input[7].SetYesNo(0);
		sc.Input[8].Name="historical data";
		sc.Input[8].SetYesNo(0);
		sc.Input[9].Name="SECONDS_PER_MINUTE";
		sc.Input[9].SetInt(1);
		sc.Input[10].Name = "Interpolation";
		sc.Input[10].SetYesNo(1);
		SYMBOL1.Name = "SYMBOL1_Interpolation";
        SYMBOL1.SetStudyID(0);
        SYMBOL1Subgraph.Name = "Price type for symbol 1";
        SYMBOL1Subgraph.SetSubgraphIndex(3);

        SYMBOL2.Name = "SYMBOL2_Interpolation";
        SYMBOL2.SetStudyID(0);
        SYMBOL2Subgraph.Name = "Price type for symbol 2";
        SYMBOL2Subgraph.SetSubgraphIndex(3);
		
		sim2.Name = "the result2 of similarity";
		sim2.PrimaryColor = RGB(0,255,0);
		sim2.DrawStyle = DRAWSTYLE_LINE;

		// Set the name of the first subgraph
		sim.Name = "the result1 of similarity";

		// Set the color and style of the subgraph line.  
		sim.PrimaryColor = RGB(255,0,0);  // Red
		sim.DrawStyle = DRAWSTYLE_LINE;
		
		// Must return before doing any data processing if sc.SetDefaults is set
		return;
	}



	//get the data of another symbol
	int& ChartNumber = sc.GetPersistentInt(1);
	int& ChartNumber2 = sc.GetPersistentInt(2);
	int threads_count =  sc.Input[1].GetInt();
	// bool sequence_normalization = sc.Input[2].GetYesNo();
	// int  mean_len = sc.Input[3].GetInt();
	// int  standard_deviation_length = sc.Input[4].GetInt();
	int window_size = sc.Input[5].GetInt();
	bool second_symbol = sc.Input[7].GetYesNo();
	bool historiy = sc.Input[8].GetYesNo();
	int minutes = sc.Input[9].GetInt();
	bool Interpolation = sc.Input[10].GetYesNo();


	//data check

	if(threads_count<=1){
		threads_count = 1;
		sc.Input[1].SetInt(threads_count);
	}
	else if(threads_count>20){
		threads_count = 20;
		sc.Input[1].SetInt(threads_count);
	}

	// if(mean_len<=10){
	// 	mean_len = 10;
	// 	sc.Input[3].SetInt(mean_len);
	// }
	// else if(mean_len>100){
	// 	mean_len = 100;
	// 	sc.Input[3].SetInt(mean_len);
	// }

	// if(standard_deviation_length<=10){
	// 	standard_deviation_length = 10;
	// 	sc.Input[4].SetInt(standard_deviation_length);
	// }
	// else if(standard_deviation_length>100){
	// 	standard_deviation_length = 100;
	// 	sc.Input[4].SetInt(standard_deviation_length);
	// }

	if(window_size<=7){
		window_size = 7;
		sc.Input[5].SetInt(window_size);
	}
	else if(window_size>=5000){
		window_size = 500;
		sc.Input[5].SetInt(window_size);
	}
	


	int ChartNumberArray[2];
	if (sc.IsFullRecalculation && !Interpolation)
	{
		GetSymbolData(ChartNumberArray ,sc.Input[0].GetString(),minutes,historiy,second_symbol,sc.Input[6].GetString(),ChartNumber,sc);
		ChartNumber = ChartNumberArray[0];
		ChartNumber2 = ChartNumberArray[1];
		
		
		

	}


	if ((ChartNumber != 0 && ChartNumber2!=0)||(ChartNumber!=0 && !second_symbol) || Interpolation)
	{
		
		SCFloatArray S1Array,S2Array;

		SCGraphData ReferenceChartData;
		SCGraphData ReferenceChartData2;

		// Get the arrays from the reference chart
		if(!Interpolation){
			sc.GetChartBaseData(ChartNumber, ReferenceChartData);
			S1Array = ReferenceChartData[SC_LAST];
			if(second_symbol){
				sc.GetChartBaseData(ChartNumber2, ReferenceChartData2);
				S2Array = ReferenceChartData2[SC_LAST];
			}
		}
		else{
			
			sc.GetStudyArrayUsingID(SYMBOL1.GetStudyID(),SYMBOL1Subgraph.GetSubgraphIndex(),S1Array);

			if(second_symbol){
				sc.GetStudyArrayUsingID(SYMBOL2.GetStudyID(),SYMBOL2Subgraph.GetSubgraphIndex(),S2Array);
			}
		}
		

		
		if ((S1Array.GetArraySize() > 0 &&S2Array.GetArraySize() > 0)||(S1Array.GetArraySize() > 0 && !second_symbol))
		{
			
			
			// int sequence_normalization_start = 0;
			// if(!sequence_normalization){
			// 	int len_x=sc.ArraySize-1;
			// 	int len_y = S1Array.GetArraySize()-1;
			// 	for(int i=0;i<len_x;i++){
			// 	sim.Arrays[0][i] = sc.BaseData[SC_LAST][i+1]-sc.BaseData[SC_LAST][i];
			// 	sim.Arrays[1][i] = S1Array[i+1]-S1Array[i];
			// 	if(second_symbol){
			// 		sim.Arrays[2][i] = S2Array[i+1]-S2Array[i];
			// 	}
				
			// 	}
			// }
			// else{

			// 	sequence_normalization_start = mean_len-1;
			// 	if((standard_deviation_length-1)>sequence_normalization_start){
			// 		sequence_normalization_start = standard_deviation_length-1;
			// 	}

			// 	#pragma omp parallel for num_threads(threads_count)
			// 	for(int i = sequence_normalization_start;i<sc.ArraySize;i++){

			// 		sim.Arrays[0][i] = (sc.BaseData[SC_LAST][i]-SMA(sc.BaseData[SC_LAST],mean_len,i))/Standard_Deviation(sc.BaseData[SC_LAST],standard_deviation_length,i);
			// 		sim.Arrays[1][i] = (S1Array[i]-SMA(S1Array,mean_len,i))/Standard_Deviation(S1Array,standard_deviation_length,i);
			// 		if(second_symbol){
			// 			sim.Arrays[2][i] = (S2Array[i]-SMA(S2Array,mean_len,i))/Standard_Deviation(S2Array,standard_deviation_length,i);
			// 		}
					
			// 	}

			// }
			


			sc.DataStartIndex = 0;
			#pragma omp parallel for num_threads(threads_count)
			for (int BarIndex = sc.UpdateStartIndex; BarIndex < sc.ArraySize; BarIndex++){
				
				//data1[BarIndex] = sim.Arrays[0][BarIndex];
				//data2[BarIndex] = sim.Arrays[1][BarIndex];
				//data[BarIndex] = ReferenceChartData[SC_LAST][BarIndex];
				
				if(BarIndex>=window_size-1 ){
					
					double res = DWT(sc.BaseData[SC_LAST],S1Array,window_size,window_size,BarIndex);
					sim[BarIndex] =  (float)res;

					if(second_symbol){
						double res2 = DWT(sc.BaseData[SC_LAST],S2Array,window_size,window_size,BarIndex);					
						sim2[BarIndex] = (float)res2;
					}
					
				}
				
				
			}
				
				// Subgraph_Output[BarIndex] = 7;
		}
		else{
			sc.AddMessageToLog("the length of the data of the new symbol is 0",1);
		}
	}
	else{
			sc.AddMessageToLog("error happened when opening the data of the new symbol",1);
	}
	
}

SCSFExport scsf_Euclidean_dis(SCStudyInterfaceRef sc){
	SCSubgraphRef sim = sc.Subgraph[0];
	SCSubgraphRef sim2 = sc.Subgraph[1];
	SCInputRef SYMBOL1 = sc.Input[11];
	SCInputRef SYMBOL1Subgraph = sc.Input[12];
	SCInputRef SYMBOL2 = sc.Input[13];
	SCInputRef SYMBOL2Subgraph = sc.Input[14];

    if (sc.SetDefaults)  
	{
		sc.GraphName = "Euclidean_dis";
		
		sc.StudyDescription = "This function is an implementation of the Euclidean_dis using openmp ";
		
		sc.AutoLoop = 0;  // true
		sc.FreeDLL = 1;
		sc.CalculationPrecedence = LOW_PREC_LEVEL;

		
		// Set the Chart Region to draw the graph in.  Region 0 is the main
		// price graph region.
		sc.GraphRegion = 1;

		sc.Input[0].Name = "SYMBOL";
		sc.Input[0].SetString(sc.GetRealTimeSymbol());
		sc.Input[1].Name = "num_threads";
		sc.Input[1].SetInt(8);
		// sc.Input[2].Name="Sequence normalization";
		// sc.Input[2].SetYesNo(1);
		// sc.Input[3].Name = "mean length for sequence normalization ";
		// sc.Input[3].SetInt(10);
		// sc.Input[4].Name = "standard_deviation length for sequence normalization";
		// sc.Input[4].SetInt(10);
		sc.Input[5].Name = "similarity window size ";
		sc.Input[5].SetInt(300);
		sc.Input[6].Name = "SYMBOL2";
		sc.Input[6].SetString(sc.GetRealTimeSymbol());
		sc.Input[7].Name="Second Symbol";
		sc.Input[7].SetYesNo(0);
		sc.Input[8].Name="historical data";
		sc.Input[8].SetYesNo(0);
		sc.Input[9].Name="SECONDS_PER_MINUTE";
		sc.Input[9].SetInt(1);
		sc.Input[10].Name = "Interpolation";
		sc.Input[10].SetYesNo(1);
		SYMBOL1.Name = "SYMBOL1_Interpolation";
	    SYMBOL1.SetStudyID(0);
	    SYMBOL1Subgraph.Name = "Price type for symbol 1";
	    SYMBOL1Subgraph.SetSubgraphIndex(3);

	    SYMBOL2.Name = "SYMBOL2_Interpolation";
	    SYMBOL2.SetStudyID(0);
	    SYMBOL2Subgraph.Name = "Price type for symbol 2";
	    SYMBOL2Subgraph.SetSubgraphIndex(3);


		
		sim2.Name = "the result2 of similarity";
		sim2.PrimaryColor = RGB(0,255,0);
		sim2.DrawStyle = DRAWSTYLE_LINE;

		// Set the name of the first subgraph
		sim.Name = "the result1 of similarity";

		// Set the color and style of the subgraph line.  
		sim.PrimaryColor = RGB(255,0,0);  // Red
		sim.DrawStyle = DRAWSTYLE_LINE;
		
		// Must return before doing any data processing if sc.SetDefaults is set
		return;
	}



	//get the data of another symbol
	int& ChartNumber = sc.GetPersistentInt(1);
	int& ChartNumber2 = sc.GetPersistentInt(2);
	int threads_count =  sc.Input[1].GetInt();
	// bool sequence_normalization = sc.Input[2].GetYesNo();
	// int  mean_len = sc.Input[3].GetInt();
	// int  standard_deviation_length = sc.Input[4].GetInt();
	int window_size = sc.Input[5].GetInt();
	bool second_symbol = sc.Input[7].GetYesNo();
	bool historiy = sc.Input[8].GetYesNo();
	int minutes = sc.Input[9].GetInt();
	bool Interpolation = sc.Input[10].GetYesNo();


	//data check

	if(threads_count<=1){
		threads_count = 1;
		sc.Input[1].SetInt(threads_count);
	}
	else if(threads_count>20){
		threads_count = 20;
		sc.Input[1].SetInt(threads_count);
	}

	// if(mean_len<=10){
	// 	mean_len = 10;
	// 	sc.Input[3].SetInt(mean_len);
	// }
	// else if(mean_len>100){
	// 	mean_len = 100;
	// 	sc.Input[3].SetInt(mean_len);
	// }

	// if(standard_deviation_length<=10){
	// 	standard_deviation_length = 10;
	// 	sc.Input[4].SetInt(standard_deviation_length);
	// }
	// else if(standard_deviation_length>100){
	// 	standard_deviation_length = 100;
	// 	sc.Input[4].SetInt(standard_deviation_length);
	// }

	if(window_size<=7){
		window_size = 7;
		sc.Input[5].SetInt(window_size);
	}
	else if(window_size>=500){
		window_size = 500;
		sc.Input[5].SetInt(window_size);
	}
	

	int ChartNumberArray[2];

	if (sc.IsFullRecalculation)
	{
		GetSymbolData(ChartNumberArray ,sc.Input[0].GetString(),minutes,historiy,second_symbol,sc.Input[6].GetString(),ChartNumber,sc);
		ChartNumber = ChartNumberArray[0];
		ChartNumber2 = ChartNumberArray[1];
		

	}


	if ((ChartNumber != 0 && ChartNumber2!=0)||(ChartNumber!=0 && !second_symbol)|| Interpolation)
	{
		
		SCFloatArray S1Array,S2Array;

		SCGraphData ReferenceChartData;
		SCGraphData ReferenceChartData2;

		// Get the arrays from the reference chart
		if(!Interpolation){
			sc.GetChartBaseData(ChartNumber, ReferenceChartData);
			S1Array = ReferenceChartData[SC_LAST];
			if(second_symbol){
				sc.GetChartBaseData(ChartNumber2, ReferenceChartData2);
				S2Array = ReferenceChartData2[SC_LAST];
			}
		}
		else{
			
			sc.GetStudyArrayUsingID(SYMBOL1.GetStudyID(),SYMBOL1Subgraph.GetSubgraphIndex(),S1Array);

			if(second_symbol){
				sc.GetStudyArrayUsingID(SYMBOL2.GetStudyID(),SYMBOL2Subgraph.GetSubgraphIndex(),S2Array);
			}
		}
		
		if ((S1Array.GetArraySize() > 0 &&S2Array.GetArraySize() > 0)||(S1Array.GetArraySize() > 0 && !second_symbol))
		{
			
			
			sc.DataStartIndex = 0;
			#pragma omp parallel for num_threads(threads_count)
			for (int BarIndex = sc.UpdateStartIndex; BarIndex < sc.ArraySize; BarIndex++){
				
				//data1[BarIndex] = sim.Arrays[0][BarIndex];
				//data2[BarIndex] = sim.Arrays[1][BarIndex];
				//data[BarIndex] = ReferenceChartData[SC_LAST][BarIndex];
				if(BarIndex>=window_size-1){
					double res = Euclidean_dis(sc.BaseData[SC_LAST],S1Array,window_size,BarIndex);
					sim[BarIndex] =  (float)res;

					if(second_symbol){
						double res2 = Euclidean_dis(sc.BaseData[SC_LAST],S2Array,window_size,BarIndex);					
						sim2[BarIndex] = (float)res2;
					}
					
				}
				
			}
				
				// Subgraph_Output[BarIndex] = 7;
		}
		else{
			sc.AddMessageToLog("the length of the data of the new symbol is 0",1);
		}
	}
	else{
			sc.AddMessageToLog("error happened when opening the data of the new symbol",1);
	}
	
}


SCSFExport scsf_normalization_display(SCStudyInterfaceRef sc){
	SCSubgraphRef d1 = sc.Subgraph[0];
	SCSubgraphRef d2 = sc.Subgraph[1];
	SCSubgraphRef d3 = sc.Subgraph[2];
	SCInputRef SYMBOL1 = sc.Input[11];
	SCInputRef SYMBOL1Subgraph = sc.Input[12];
	SCInputRef SYMBOL2 = sc.Input[13];
	SCInputRef SYMBOL2Subgraph = sc.Input[14];

    if (sc.SetDefaults)  
	{
		sc.GraphName = "normalization_display";
		
		sc.StudyDescription = "display the result of normalization. ";
		
		sc.AutoLoop = 0;  // true
		sc.FreeDLL = 1;
		sc.CalculationPrecedence = LOW_PREC_LEVEL;

		
		// Set the Chart Region to draw the graph in.  Region 0 is the main
		// price graph region.
		sc.GraphRegion = 1;

		sc.Input[0].Name = "SYMBOL";
		sc.Input[0].SetString(sc.GetRealTimeSymbol());
		sc.Input[1].Name = "num_threads";
		sc.Input[1].SetInt(1);
		sc.Input[3].Name = "mean length for sequence normalization ";
		sc.Input[3].SetInt(10);
		sc.Input[4].Name = "standard_deviation length for sequence normalization";
		sc.Input[4].SetInt(10);
		sc.Input[6].Name = "SYMBOL2";
		sc.Input[6].SetString(sc.GetRealTimeSymbol());
		sc.Input[7].Name="Second Symbol";
		sc.Input[7].SetYesNo(1);
		sc.Input[8].Name="historical data";
		sc.Input[8].SetYesNo(0);
		sc.Input[9].Name="SECONDS_PER_MINUTE";
		sc.Input[9].SetInt(1);
		sc.Input[10].Name = "Interpolation";
		sc.Input[10].SetYesNo(1);
		SYMBOL1.Name = "SYMBOL1_Interpolation";
	    SYMBOL1.SetStudyID(0);
	    SYMBOL1Subgraph.Name = "Price type for symbol 1";
	    SYMBOL1Subgraph.SetSubgraphIndex(3);

	    SYMBOL2.Name = "SYMBOL2_Interpolation";
	    SYMBOL2.SetStudyID(0);
	    SYMBOL2Subgraph.Name = "Price type for symbol 2";
	    SYMBOL2Subgraph.SetSubgraphIndex(3);

		
		d1.Name = "the normalization of original symbol ";
		d1.PrimaryColor = RGB(0,255,0);
		d1.DrawStyle = DRAWSTYLE_LINE;

		// Set the name of the first subgraph
		d2.Name = "the normalization of the first symbol ";
		d2.PrimaryColor = RGB(255,0,0);  // Red
		d2.DrawStyle = DRAWSTYLE_LINE;

		d3.Name="the normalization of the second symbol";
		d2.PrimaryColor = RGB(0,0,255);
		d3.DrawStyle = DRAWSTYLE_LINE;
		
		// Must return before doing any data processing if sc.SetDefaults is set
		return;
	}



	//get the data of another symbol
	int& ChartNumber = sc.GetPersistentInt(1);
	int& ChartNumber2 = sc.GetPersistentInt(2);
	int threads_count =  sc.Input[1].GetInt();
	int  mean_len = sc.Input[3].GetInt();
	int  standard_deviation_length = sc.Input[4].GetInt();
	bool second_symbol = sc.Input[7].GetYesNo();
	bool historiy = sc.Input[8].GetYesNo();
	int minutes = sc.Input[9].GetInt();
	bool Interpolation = sc.Input[10].GetYesNo();


	//data check

	if(threads_count<1){
		threads_count = 1;
		sc.Input[1].SetInt(threads_count);
	}
	else if(threads_count>20){
		threads_count = 20;
		sc.Input[1].SetInt(threads_count);
	}

	if(mean_len<=10){
		mean_len = 10;
		sc.Input[3].SetInt(mean_len);
	}
	else if(mean_len>100){
		mean_len = 100;
		sc.Input[3].SetInt(mean_len);
	}

	if(standard_deviation_length<10){
		standard_deviation_length = 10;
		sc.Input[4].SetInt(standard_deviation_length);
	}
	else if(standard_deviation_length>100){
		standard_deviation_length = 100;
		sc.Input[4].SetInt(standard_deviation_length);
	}

	


	int ChartNumberArray[2];
	if (sc.IsFullRecalculation)
	{
		GetSymbolData(ChartNumberArray ,sc.Input[0].GetString(),minutes,historiy,second_symbol,sc.Input[6].GetString(),ChartNumber,sc);
		ChartNumber = ChartNumberArray[0];
		ChartNumber2 = ChartNumberArray[1];
		

	}


	if ((ChartNumber != 0 && ChartNumber2!=0)||(ChartNumber!=0 && !second_symbol)||!Interpolation)
	{
		
		
		SCFloatArray S1Array,S2Array;

		SCGraphData ReferenceChartData;
		SCGraphData ReferenceChartData2;

		// Get the arrays from the reference chart
		if(!Interpolation){
			sc.GetChartBaseData(ChartNumber, ReferenceChartData);
			S1Array = ReferenceChartData[SC_LAST];
			if(second_symbol){
				sc.GetChartBaseData(ChartNumber2, ReferenceChartData2);
				S2Array = ReferenceChartData2[SC_LAST];
			}
		}
		else{
			
			sc.GetStudyArrayUsingID(SYMBOL1.GetStudyID(),SYMBOL1Subgraph.GetSubgraphIndex(),S1Array);

			if(second_symbol){
				sc.GetStudyArrayUsingID(SYMBOL2.GetStudyID(),SYMBOL2Subgraph.GetSubgraphIndex(),S2Array);
			}
		}




		if ((S1Array.GetArraySize() > 0 &&S2Array.GetArraySize() > 0)||(S1Array.GetArraySize() > 0 && !second_symbol))
		{
			

			int sequence_normalization_start;
			sequence_normalization_start = mean_len-1;
			if((standard_deviation_length-1)>sequence_normalization_start){
				sequence_normalization_start = standard_deviation_length-1;
			}

			sc.DataStartIndex = sequence_normalization_start;
			#pragma omp parallel for num_threads(threads_count)
			for(int i = sc.UpdateStartIndex;i<sc.ArraySize;i++){
				if(i>=sequence_normalization_start){
					float res1 = (sc.BaseData[SC_LAST][i]-SMA(sc.BaseData[SC_LAST],mean_len,i))/Standard_Deviation(sc.BaseData[SC_LAST],standard_deviation_length,i);
					float res2 = (S1Array[i]-SMA(S1Array,mean_len,i))/Standard_Deviation(S1Array,standard_deviation_length,i);
					d1[i] = res1;
					d2[i] = res2;
					if(second_symbol){
						float res3= (S2Array[i]-SMA(S2Array,mean_len,i))/Standard_Deviation(S2Array,standard_deviation_length,i);
						d3[i] = res3;
					}
						
					
				}
				
			}		
				
		}
		else{
			sc.AddMessageToLog("the length of the data of the new symbol is 0",1);
		}
	}
	else{
			sc.AddMessageToLog("error happened when opening the data of the new symbol",1);
	}
	
}

SCSFExport scsf_test_time(SCStudyInterfaceRef sc)
{
	// Section 1 - Set the configuration variables and defaults
	
	
	if (sc.SetDefaults)  
	{
		sc.GraphName = "time test";
		
		sc.StudyDescription = "Example function for calculating a simple moving average from scratch.";
		
		sc.AutoLoop = 0;  // fasle
		sc.FreeDLL = 1;
		
		// Set the Chart Region to draw the graph in.  Region 0 is the main
		// price graph region.
		sc.GraphRegion = 0;
		
		// Set the name of the first subgraph
		sc.Subgraph[0].Name = "distance";

		// Set the color and style of the subgraph line.  
		sc.Subgraph[0].PrimaryColor = RGB(0,0,255);  // Red
		sc.Subgraph[0].DrawStyle = DRAWSTYLE_LINE;
		
		
		
		// Must return before doing any data processing if sc.SetDefaults is set
		return;
	}
	// char alert[20];
	// sprintf_s(alert, "%d", ReferenceChartData[SC_LAST].GetArraySize());
	// sc.AddMessageToLog(SCString(alert) ,1);
	
	SCDateTime time1;
	
	sc.DataStartIndex = 0;
	for(int i=sc.UpdateStartIndex;i<5;i++){
		time1 = sc.BaseDateTimeIn[i];
		char alert[20] ,hour[5],minute[5];
		int temp;
		temp = time1.GetHour();
		sprintf_s(hour,"%d",temp);
		temp = time1.GetMinute();
		sprintf_s(minute,"%d",temp);
		strcpy(alert,hour);
		strcat(alert," h ");
		strcat(alert,minute);
		
		//sprintf_s(alert, "%d", temp);
		
		sc.AddMessageToLog(SCString(alert) ,1);

		// time2 = sc.BaseDataEndDateTime[i];
		// temp = time2.GetMinute();
		// sprintf_s(alert, "%d", temp);
		// sc.AddMessageToLog(SCString(alert) ,1);
	}
	
}

SCSFExport scsf_StringExamples(SCStudyInterfaceRef sc)
{
	if (sc.SetDefaults)
	{
		// Set the configuration and defaults
		
		sc.GraphName = "String Examples";
		
		sc.StudyDescription = "Working with Strings Examples.";

		
		
		
		return;
	}
	
	
	// Do data processing
	
	// Comparison Example
	SCString Message;
	int ReturnValue;
	Message.Format("This here is a Test");
	ReturnValue = Message.CompareNoCase("This is a test");

	// Direct String Access Example
	const char* p_Symbol;
	p_Symbol = sc.Symbol.GetChars();


	SCString Left = Message.Left(4);
	SCString Right = Message.Right(4);

	Message = "Left: ";
	Message += Left;
	Message += ", Right: ";
	Message += Right;
	sc.AddMessageToLog(Message, 1);

	Message.AppendFormat(", Length: %d", Message.GetLength());
	sc.AddMessageToLog(Message, 0);

	Message += ", DateTimeMS: ";
	Message += sc.FormatDateTimeMS(sc.CurrentSystemDateTimeMS);
	sc.AddMessageToLog(Message, 0);
}


SCSFExport scsf_show_length(SCStudyInterfaceRef sc)
{
	if (sc.SetDefaults)
	{
		// Set the configuration and defaults
		
		sc.GraphName = "sequence length";
		
		sc.StudyDescription = "show length";

		sc.AutoLoop = 0;  // fasle
		sc.FreeDLL = 1;
		sc.GraphRegion = 0;

		
		
		
		return;
	}


	char alert[20];
	sprintf_s(alert, "%d", sc.ArraySize);
	sc.AddMessageToLog(SCString(alert),1);
	
	
}



SCSFExport scsf_diff(SCStudyInterfaceRef sc){
	SCSubgraphRef data = sc.Subgraph[0];

    if (sc.SetDefaults)  
	{
		sc.GraphName = "diff";
		
		sc.StudyDescription = "display the result of normalization. ";
		
		sc.AutoLoop = 0;  // true
		sc.FreeDLL = 1;
		
		// Set the Chart Region to draw the graph in.  Region 0 is the main
		// price graph region.
		sc.GraphRegion = 1;

		sc.Input[0].Name = "SYMBOL";
		sc.Input[0].SetString(sc.GetRealTimeSymbol());
		sc.Input[8].Name="historical data";
		sc.Input[8].SetYesNo(0);
		sc.Input[9].Name="SECONDS_PER_MINUTE";
		sc.Input[9].SetInt(1);
		
		data.Name = "the normalization of original symbol ";
		data.PrimaryColor = RGB(0,255,0);
		data.DrawStyle = DRAWSTYLE_LINE;


		
		// Must return before doing any data processing if sc.SetDefaults is set
		return;
	}



	//get the data of another symbol
	int& ChartNumber = sc.GetPersistentInt(1);
	bool historiy = sc.Input[8].GetYesNo();
	int minutes = sc.Input[9].GetInt();


	//data check

	
	

	int ChartNumberArray[2];

	if (sc.IsFullRecalculation)
	{
		GetSymbolData(ChartNumberArray ,sc.Input[0].GetString(),minutes,historiy,false,NULL,ChartNumber,sc);
		ChartNumber = ChartNumberArray[0];
		

	}


	if (ChartNumber != 0)
	{
		
		SCGraphData ReferenceChartData;
		SCDateTimeArray DateTimeArray;
	

		// Get the arrays from the reference chart
		sc.GetChartBaseData(ChartNumber, ReferenceChartData);
		sc.GetChartDateTimeArray(ChartNumber, DateTimeArray);
		if (ReferenceChartData[SC_LAST].GetArraySize() > 0 )
		{
			
			sc.DataStartIndex = 0;

			SCDateTime t1,t2;
			int h1,h2,m1,m2,d1,d2,y1,y2,month1,month2;
			for(int i = sc.UpdateStartIndex;i<sc.ArraySize;i++){
				t1 = sc.BaseDateTimeIn[i];
				t2 = DateTimeArray[i];

				y1 = t1.GetYear();
				month1 = t1.GetMonth();
				d1 = t1.GetDay();
				h1 = t1.GetHour();
				m1 = t1.GetMinute();

				y2 = t2.GetYear();
				month2 = t2.GetMonth();
				d2 = t2.GetDay();
				h2 = t2.GetHour();
				m2 = t2.GetMinute();
				if(d1==d2){
					if(h1==h2){
						if(m1==m2){

						}
						else{
							char temp1[5],temp2[5],temp3[5],temp4[5],temp5[5],alert[20];
							sprintf_s(temp1,"%d",d2);
							sprintf_s(temp2,"%d",h2);
							sprintf_s(temp3,"%d",m2);
							sprintf_s(temp4,"%d",month2);
							sprintf_s(temp5,"%d",y2);
							strcpy(alert,temp5);
							strcat(alert,"y");
							strcat(alert,temp4);
							strcat(alert,"month");
							strcat(alert,temp1);
							strcat(alert,"d");
							strcat(alert,temp2);
							strcat(alert,"h");
							strcat(alert,temp3);

							sc.AddMessageToLog(SCString(alert),1);
							break;
						}
					}
				}
		
			}		
				
		}
		else{
			sc.AddMessageToLog("the length of the data of the new symbol is 0",1);
		
		}
		

	}
	else{
			sc.AddMessageToLog("error happened when opening the data of the new symbol",1);
	}
}



SCSFExport scsf_similarity_dtw_parallel(SCStudyInterfaceRef sc){
	SCSubgraphRef sim = sc.Subgraph[0];
	SCSubgraphRef sim2 = sc.Subgraph[1];
	SCInputRef SYMBOL1 = sc.Input[11];
    SCInputRef SYMBOL1Subgraph = sc.Input[12];
    SCInputRef SYMBOL2 = sc.Input[13];
    SCInputRef SYMBOL2Subgraph = sc.Input[14];

    if (sc.SetDefaults)  
	{
		sc.GraphName = "dtw_parallel";
		
		sc.StudyDescription = "This function is an implementation of the dtw function using openmp ";
		
		sc.AutoLoop = 0;  // true
		sc.FreeDLL = 1;
		sc.CalculationPrecedence = LOW_PREC_LEVEL;
		
		// Set the Chart Region to draw the graph in.  Region 0 is the main
		// price graph region.
		sc.GraphRegion = 1;

		sc.Input[0].Name = "SYMBOL";
		sc.Input[0].SetString(sc.GetRealTimeSymbol());
		sc.Input[1].Name = "num_threads";
		sc.Input[1].SetInt(8);
		sc.Input[2].Name="Sequence normalization";
		sc.Input[5].Name = "similarity window size ";
		sc.Input[5].SetInt(300);
		sc.Input[6].Name = "SYMBOL2";
		sc.Input[6].SetString(sc.GetRealTimeSymbol());
		sc.Input[7].Name="Second Symbol";
		sc.Input[7].SetYesNo(0);
		sc.Input[8].Name="historical data";
		sc.Input[8].SetYesNo(0);
		sc.Input[9].Name="SECONDS_PER_MINUTE";
		sc.Input[9].SetInt(1);
		sc.Input[10].Name = "Interpolation";
		sc.Input[10].SetYesNo(1);
		SYMBOL1.Name = "SYMBOL1_Interpolation";
        SYMBOL1.SetStudyID(0);
        SYMBOL1Subgraph.Name = "Price type for symbol 1";
        SYMBOL1Subgraph.SetSubgraphIndex(3);

        SYMBOL2.Name = "SYMBOL2_Interpolation";
        SYMBOL2.SetStudyID(0);
        SYMBOL2Subgraph.Name = "Price type for symbol 2";
        SYMBOL2Subgraph.SetSubgraphIndex(3);
		
		sim2.Name = "the result2 of similarity";
		sim2.PrimaryColor = RGB(0,255,0);
		sim2.DrawStyle = DRAWSTYLE_LINE;

		// Set the name of the first subgraph
		sim.Name = "the result1 of similarity";

		// Set the color and style of the subgraph line.  
		sim.PrimaryColor = RGB(255,0,0);  // Red
		sim.DrawStyle = DRAWSTYLE_LINE;
		
		// Must return before doing any data processing if sc.SetDefaults is set
		return;
	}



	//get the data of another symbol
	int& ChartNumber = sc.GetPersistentInt(1);
	int& ChartNumber2 = sc.GetPersistentInt(2);
	int threads_count =  sc.Input[1].GetInt();
	
	int window_size = sc.Input[5].GetInt();
	bool second_symbol = sc.Input[7].GetYesNo();
	bool historiy = sc.Input[8].GetYesNo();
	int minutes = sc.Input[9].GetInt();
	bool Interpolation = sc.Input[10].GetYesNo();


	//data check

	if(threads_count<=1){
		threads_count = 1;
		sc.Input[1].SetInt(threads_count);
	}
	else if(threads_count>20){
		threads_count = 20;
		sc.Input[1].SetInt(threads_count);
	}


	if(window_size<=7){
		window_size = 7;
		sc.Input[5].SetInt(window_size);
	}
	else if(window_size>=5000){
		window_size = 500;
		sc.Input[5].SetInt(window_size);
	}
	


	int ChartNumberArray[2];
	if (sc.IsFullRecalculation && !Interpolation)
	{
		GetSymbolData(ChartNumberArray ,sc.Input[0].GetString(),minutes,historiy,second_symbol,sc.Input[6].GetString(),ChartNumber,sc);
		ChartNumber = ChartNumberArray[0];
		ChartNumber2 = ChartNumberArray[1];
		
		
		

	}


	if ((ChartNumber != 0 && ChartNumber2!=0)||(ChartNumber!=0 && !second_symbol) || Interpolation)
	{
		
		SCFloatArray S1Array,S2Array;

		SCGraphData ReferenceChartData;
		SCGraphData ReferenceChartData2;

		// Get the arrays from the reference chart
		if(!Interpolation){
			sc.GetChartBaseData(ChartNumber, ReferenceChartData);
			S1Array = ReferenceChartData[SC_LAST];
			if(second_symbol){
				sc.GetChartBaseData(ChartNumber2, ReferenceChartData2);
				S2Array = ReferenceChartData2[SC_LAST];
			}
		}
		else{
			
			sc.GetStudyArrayUsingID(SYMBOL1.GetStudyID(),SYMBOL1Subgraph.GetSubgraphIndex(),S1Array);

			if(second_symbol){
				sc.GetStudyArrayUsingID(SYMBOL2.GetStudyID(),SYMBOL2Subgraph.GetSubgraphIndex(),S2Array);
			}
		}
		

		
		if ((S1Array.GetArraySize() > 0 &&S2Array.GetArraySize() > 0)||(S1Array.GetArraySize() > 0 && !second_symbol))
		{
			
			
			//omp_set_nested(1);

			sc.DataStartIndex = 0;
			
			//#pragma omp parallel for num_threads(threads_count)
			for (int BarIndex = sc.UpdateStartIndex; BarIndex < sc.ArraySize; BarIndex++){
				
				//data1[BarIndex] = sim.Arrays[0][BarIndex];
				//data2[BarIndex] = sim.Arrays[1][BarIndex];
				//data[BarIndex] = ReferenceChartData[SC_LAST][BarIndex];
				
				if(BarIndex>=window_size-1 ){
					

					
					double res = dtw_parallel(sc.BaseData[SC_LAST], S1Array, window_size,BarIndex,sc);
					sim[BarIndex] =  (float)res;


					if(second_symbol){
						double res2 = dtw_parallel(sim.Arrays[0], S2Array, window_size,BarIndex,sc);					
						sim2[BarIndex] = (float)res2;
					}
					
				}
				
				
			}
				
				// Subgraph_Output[BarIndex] = 7;
		}
		else{
			sc.AddMessageToLog("the length of the data of the new symbol is 0",1);
		}
	}
	else{
			sc.AddMessageToLog("error happened when opening the data of the new symbol",1);
	}
	
}


SCSFExport scsf_search_similarity(SCStudyInterfaceRef sc){
	SCSubgraphRef sim = sc.Subgraph[0];
	SCInputRef SYMBOL1 = sc.Input[0];
    SCInputRef SYMBOL1Subgraph = sc.Input[1];
    if(sc.SetDefaults){
    	sc.GraphName = "search similarity duration";
		
		sc.StudyDescription = "This function is an implementation to search the most similar duration";
		
		sc.AutoLoop = 0;  // true
		sc.FreeDLL = 1;
		sc.CalculationPrecedence = LOW_PREC_LEVEL;
		
		// Set the Chart Region to draw the graph in.  Region 0 is the main
		// price graph region.
		sc.GraphRegion = 0;

	
		sc.Input[4].Name = "num_threads";
		sc.Input[4].SetInt(8);
		sc.Input[5].Name = "similarity window size ";
		sc.Input[5].SetInt(300);
		
		sc.Input[8].Name="historical data";
		sc.Input[8].SetYesNo(0);
		sc.Input[9].Name="SECONDS_PER_MINUTE";
		sc.Input[9].SetInt(1);
		
		SYMBOL1.Name = "SYMBOL1";
        SYMBOL1.SetStudyID(0);
        SYMBOL1Subgraph.Name = "Price type for symbol 1";
        SYMBOL1Subgraph.SetSubgraphIndex(3);

		
		// Must return before doing any data processing if sc.SetDefaults is set
		return;
    }


    int& ChartNumber = sc.GetPersistentInt(1);
	int threads_count =  sc.Input[4].GetInt();
	
	int window_size = sc.Input[5].GetInt();
	bool historiy = sc.Input[8].GetYesNo();
	int minutes = sc.Input[9].GetInt();

    if(threads_count<=1){
		threads_count = 1;
		sc.Input[1].SetInt(threads_count);
	}
	else if(threads_count>20){
		threads_count = 20;
		sc.Input[1].SetInt(threads_count);
	}


	if(window_size<=7){
		window_size = 7;
		sc.Input[5].SetInt(window_size);
	}
	else if(window_size>=5000){
		window_size = 5000;
		sc.Input[5].SetInt(window_size);
	}


	SCFloatArray S1Array;

	

	// Get the arrays from the reference study
	
	
	sc.GetStudyArrayUsingID(SYMBOL1.GetStudyID(),SYMBOL1Subgraph.GetSubgraphIndex(),S1Array);
	
	

	

	if (S1Array.GetArraySize() > 0){

		// query secquence
		int length = sc.ArraySize;	
		for(int i=0;i<window_size;i++){
			sim.Arrays[0][i] = sc.BaseData[SC_LAST][length-window_size+i];
		}
		int res =  search_similarity(sim.Arrays[0], S1Array, window_size,sc.ArraySize,threads_count,sc);
		char alert[20];
		sprintf_s(alert,"%d",res);
		sc.AddMessageToLog(SCString(alert),1);




	}
	
    
}

