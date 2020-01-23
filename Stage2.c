#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

pthread_barrier_t   barrier;

typedef struct{
	int value;
	struct trans_node *next;
}trans_node;

typedef struct {
	char Storecode[20];
	int trans_no;
}transaction;

typedef struct {
	char Storecode[20];
	int freq;
	struct trans_node* next;
}item;

typedef struct{
	int threadid;
	transaction* tx_list;
	item* item_list;
	int num_trans;
}arg_list;



typedef struct{
	int itemset_size;
	struct item *next;
	int freq;
}itemset;


#define TRANS_NUM 600000
#define NUM_THREADS 16


void *freq_cnt(void *args) {
	transaction *trans_array;
	item *item_array;
	arg_list *mul_args;
	long tid;
	int start,end;
	int i,j,k,m;
	int present=0;
	m=0;
	int dist_freq; // distinct frequencies
	char item_code[20];
	trans_node* tail;
	trans_node* temp;


	mul_args = (arg_list*)args;	
	tid = mul_args->threadid;
	trans_array = mul_args->tx_list;
	item_array = mul_args->item_list;
	start=(mul_args->num_trans/NUM_THREADS)*tid;
   	end=start+(mul_args->num_trans/NUM_THREADS);
	dist_freq=start;

	printf("\nEntering thread %d\n", tid);

	//trans_node* trans_node_array = (trans_node*)malloc(sizeof(trans_node)*(mul_args->num_trans/NUM_THREADS));


	//calculating frequency
	//printf("threadid:%d start:%d, end:%d dist_freq:%d\n", tid,start,end,dist_freq);
	for(i=start;i<(end-1);i++){
		strcpy(item_code,trans_array[i].Storecode);
		for(j = start;j < dist_freq;j++){  //check whether you have already cacluated frequency
			if(strcmp(item_code, item_array[j].Storecode)==0){
				present=1;
				break;
			}		
		}

		if(!present){  //if frequency uis not calculated,calculate it
	    //add first item
			strcpy(item_array[dist_freq].Storecode,item_code);
			item_array[dist_freq].freq=1;
			/*
			trans_node_array[m].value=trans_array[i].trans_no;
			trans_node_array[m].next=NULL;
			item_array[dist_freq].next=(trans_node *)trans_node_array;//m
			tail=(trans_node *)item_array[dist_freq].next;
			m++;*/
			trans_node* temp = (trans_node*)malloc(sizeof(trans_node));
			temp->value=trans_array[i].trans_no;
			temp->next=NULL;
			item_array[dist_freq].next=temp;
			tail=(trans_node *)item_array[dist_freq].next;

			for(k=i+1;k<(end-1);k++){ //add all other items
				if(strcmp(item_code, trans_array[k].Storecode)==0){
					item_array[dist_freq].freq++;
					/*trans_node_array[m].value=trans_array[i].trans_no;
					trans_node_array[m].next=NULL;
					tail.next=(trans_node *)trans_node_array;//[m];*/
					trans_node* temp = (trans_node*)malloc(sizeof(trans_node));
					temp->value=trans_array[k].trans_no;
					temp->next=NULL;
					tail->next=(trans_node *)temp;
					tail=(trans_node *)tail->next;
					//m++;
				}
			}
			dist_freq++;
		}
		else{
			present=0;
			continue;	
		}
	}
	//printf("\nNo of distinct items:%d",dist_freq-start);
	printf("\n Frequency of items for thread %d\n",tid);
	for(i=0;i<dist_freq;i++){
		printf("\n %s %d\t",item_array[i].Storecode,item_array[i].freq);
		temp=(trans_node*)item_array[i].next;
		while(temp!=NULL){
			printf("%d\t",temp->value);
			temp=temp->next;
		}
	}
	strcpy(item_array[dist_freq].Storecode,"end_of_item_list");

	printf("\nLeaving thread %d\n", tid);
	pthread_exit(NULL);
}


int main(void) {
	int rc,j,k,l,m, present;
	int support;
    char *buf;
	char item_num[20];
    pthread_t threads[NUM_THREADS];
	transaction* trans_array = (transaction*)malloc(sizeof(transaction)*TRANS_NUM);
	item* item_array = (item*)malloc(sizeof(item)*TRANS_NUM);
	item* item_frequency = (item*)malloc(sizeof(item)*TRANS_NUM);
	item* item_frequency_min_sup = (item*)malloc(sizeof(item)*TRANS_NUM);
	arg_list *args[NUM_THREADS]; // for passing to the threads
	trans_node *temp,*temp1,*tail;
	int item_start,item_end_1,item_end_2,item_num_freq;
	item topo_temp;
	int itemset_no;
	itemset* itemset_2;
	int freq=0;
	item first;
	trans_node* ptr;
	int itemset_size,iter;
	item first_ptr;
	int match_found;
	item *f_item,*s_item;
	item *item_temp,*item_temp1;
	itemset is_temp;
	item *holder1;
	item *holder2;
	int first_value;

	temp = (trans_node *)malloc(sizeof(trans_node)); //seg_fault
	temp1 = (trans_node *)malloc(sizeof(trans_node)); //seg_fault


	//pthread_barrier_init (&barrier, NULL, NUM_THREADS+1);

	//printf("\nEnter minimum support count:");
	//scanf("%d",&support);
	support=2;


    FILE *fp = fopen("ex_data.csv", "r"); //"Online_Retail.csv
    if (!fp) {
        printf("Can't open file\n");
        return 0;
    }
	size_t len=0;
	int row_count=0;
    int i = 0;
    int field_count = 0;

//    args=(arg_list *)malloc(sizeof(arg_list));
//    while (fgets(buf, 1024, fp)) {
	while ((getline(&buf, &len, fp))!=-1) {
		field_count = 0;
        if (row_count == 1) {
            continue;
        }

        char *field = strtok(buf, ",");
        while (field) {
            if (field_count == 0) {
                trans_array[i].trans_no=atoi(field);
            }
            if (field_count == 1) {
                strcpy(trans_array[i].Storecode,field);
               //trans_array[i].Storecode=atoi(field);  
			}
			field = strtok(NULL, ",");
            field_count++;
        }
		i=i+1;
    }
    //printf("\nreading done\n");
	row_count=i;
	//printf("number of transactions:%d", row_count);
//	for(i=0;i<row_count;i++){
		//printf("iteration num %d\n",i);
//		printf("%s %d\n",trans_array[i].Storecode,trans_array[i].trans_no);
//	}

	
	fclose(fp);

	for(i=0;i<NUM_THREADS;i++){ //NUM_THREADS
		args[i]=(arg_list *)malloc(sizeof(arg_list));
		args[i]->threadid=i;
		args[i]->tx_list=trans_array;
		args[i]->item_list=item_array;
		args[i]->num_trans=row_count;
		rc=pthread_create(&threads[i], NULL, freq_cnt, args[i] );
		if (rc) {
			printf("Error:unable to create thread, %d\n", rc);
			exit(-1);
		}
	}

	//pthread_barrier_wait (&barrier);

	for(i=0;i<NUM_THREADS;i++){
		pthread_join(threads[i], NULL);
	}

	printf("\n finished individual threads");

	//each thread will produce word ciunt, Take first threads result, check whether it is in consolidated result, include if not and then check for results of all subsequentthreads, repeat for subsequent threads	
	item_start=0;item_end_1=0;
	for(i=0;i<NUM_THREADS;i++){
		j=(row_count/NUM_THREADS)*i;
		while(strcmp("end_of_item_list",item_array[j].Storecode)!=0){
			strcpy(item_num,item_array[j].Storecode);
			item_num_freq=item_array[j].freq;
			for(m = item_start;m < item_end_1;m++){  //check whether you have already cacluated frequency
				if(strcmp(item_num, item_frequency[m].Storecode)==0){
					present=1;
					break;
				}
			}

			if(!present){  //if frequency uis not calculated,calculate it
			//add first item
				strcpy(item_frequency[item_end_1].Storecode,item_num);
				item_frequency[item_end_1].freq=item_num_freq;
				item_frequency[item_end_1].next=item_array[item_end_1].next;
				temp1=(trans_node*)item_array[item_end_1].next;
				while(temp1!=NULL){
					tail=temp1;
					temp1=temp1->next;
				}
				
				for(l=i+1;l<NUM_THREADS;l++){				
					k=(row_count/NUM_THREADS)*l;
					while(strcmp("end_of_item_list",item_array[k].Storecode)!=0){  //calculate frequency if calculated elsewher
						if(strcmp(item_num, item_array[k].Storecode)==0){
							item_frequency[item_end_1].freq+=item_array[k].freq;
							temp1=(trans_node*)item_array[k].next;
							tail->next=temp1;
							//temp1=tail->next;
							while(temp1!=NULL){
								tail=temp1;
								temp1=temp1->next;
							}
							break; //it isthe only distinct value present
						}
						k++;
					}
				}
				item_end_1++;
				j++;
			}
			else{
				present=0;
				j++;
				continue;	
			}
		}
	}



	printf("\nTotal distinct words before pruning:%d",item_end_1);
//pruning items
	j=0;
	itemset_no=0;
	for(i=0;i<item_end_1;i++){
		//printf("%s %d\n",item_frequency[i].Storecode,item_frequency[i].freq);
		//temp=(trans_node*)item_frequency[i].next;
		//while(temp!=NULL){
		//	printf(" %d/t",temp->value);
		//	temp = temp->next;
		//}

		if(item_frequency[i].freq>support){
			strcpy(item_frequency_min_sup[j].Storecode,item_frequency[i].Storecode);
			item_frequency_min_sup[j].freq=item_frequency[i].freq;
			item_frequency_min_sup[j].next=item_frequency[i].next;
			j++;
			itemset_no++;
		}
	}


	item_end_1=j; //here it is the size of itemset after pruning of itemsize 1
	printf("\n total distinct words after pruning:%d\n",item_end_1);

	/*for(i=0;i<item_end_1;i++){
		printf("%s %d\n",item_frequency_min_sup[i].Storecode,item_frequency_min_sup[i].freq);
	}*/


	//lexigographic sorting
	for (i = 0; i < item_end_1; i++) {
        for (j = i + 1; j < item_end_1; j++) {
            if (strcmp(item_frequency_min_sup[i].Storecode, item_frequency_min_sup[j].Storecode) > 0) {
                topo_temp=item_frequency_min_sup[i];
                item_frequency_min_sup[i]=item_frequency_min_sup[j];
                item_frequency_min_sup[j]=topo_temp;
            }
        }
    }


    printf("\n lexigographically sorted itemset1");
	for(i=0;i<item_end_1;i++){
		printf("%s %d\n",item_frequency_min_sup[i].Storecode,item_frequency_min_sup[i].freq);
	}

	printf("\nTotal distinct words:%d",item_end_1);

	//displaying transposed database of the form Itemset -> transactions
	for(i=0;i<item_end_1;i++){
		temp=(trans_node*)item_frequency_min_sup[i].next;
		printf("\nstorecode:%s frequency:%d transaction_list: \n",item_frequency_min_sup[i].Storecode,item_frequency_min_sup[i].freq);
		while(temp!=NULL){
			printf("%d\t",temp->value);
			temp = temp->next;
		}
	}

//creating itemsets of size 2 and more
	if(itemset_no==0){
		printf("\n maximum itemset size is %d", itemset_no);
		return;
	}


	itemset_2 = (itemset *)malloc(sizeof((item_end_1*item_end_1)/2));


	//creating itemsets of size 2
	k=0;
	for(i=0;i<item_end_1;i++){
		for(j=i+1;j<item_end_1;j++){
			first = item_frequency_min_sup[i];
			ptr = first.next;
			while(ptr!=NULL){
				temp=item_frequency_min_sup[j].next;
				while(temp!=NULL){
						if(temp->value==ptr->value){
							freq++;
							break;
						}
					temp =temp->next;
				}
				if(freq>support){
					itemset_2[k].itemset_size=2;

					//first item
					holder1 = (item *)malloc(sizeof(item));
					strcpy(holder1->Storecode,item_frequency_min_sup[i].Storecode);
					holder1->freq= item_frequency_min_sup[i].freq;
					holder1->next= item_frequency_min_sup[i].next;
					itemset_2[k].next=holder1;
					
					temp1=itemset_2[k].next;

					//2nd item
					holder2 = (item *)malloc(sizeof(item));
					strcpy(holder2->Storecode,item_frequency_min_sup[j].Storecode);
					holder2->freq= item_frequency_min_sup[j].freq;
					holder2->next= item_frequency_min_sup[j].next;
					temp1->next=holder2;   //check this

					itemset_2[k].freq=freq;
				}
				ptr=ptr->next;
			}
			freq=0;
			k++;
		}
	}

	item_end_2=k; // no of elements in the itemset of size 2

	//displaying Itemset_2 -> transactions
	for(i=0;i<item_end_2;i++){
		item_temp=itemset_2[i].next;
		item_temp1=item_temp->next;
		printf("\n Item1:%s, Item2:%s, frequency:%d",item_temp->Storecode,item_temp1->Storecode,itemset_2[i].freq);
	}


	//lexicographic sorting of itemset of size 2
	for (i = 0; i < item_end_2; i++) {
        for (j = i + 1; j < item_end_2; j++) {
            f_item = itemset_2[i].next;
            s_item = itemset_2[j].next;
            if (strcmp(f_item->Storecode, s_item->Storecode) < 0) {
            	continue;
            }
            else{
            	while ((f_item!=NULL) && (s_item!=NULL)) {
            		if(strcmp(f_item->Storecode, s_item->Storecode) == 0){
	            		f_item = f_item->next;
	            		s_item = s_item->next;
            		}
            	}	
            	if((f_item!=NULL)&&(s_item!=NULL)){
		            if (strcmp(f_item->Storecode, s_item->Storecode) > 0) {
		                is_temp=itemset_2[i];
		                itemset_2[i]=itemset_2[j];
		                itemset_2[j]=is_temp;
		            }           		
            	}
            }
        }
    }

     printf("\n lexigographically sorted itemset2");
	for(i=0;i<item_end_2;i++){
		item_temp=itemset_2[i].next;
		item_temp1=item_temp->next;
		printf("\n Item1:%s, Item2:%s, frequency:%d",item_temp->Storecode,item_temp1->Storecode,itemset_2[i].freq);
	}



	pthread_exit(NULL);
	return 0;
}
