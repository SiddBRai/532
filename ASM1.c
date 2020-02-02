
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

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

typedef struct {
	char Storecode[256];
	struct plain_item* next;
}plain_item;

typedef struct{
	int threadid;
	transaction* tx_list;
	item* item_list;
	int num_trans;
}arg1_list;



typedef struct{
	int itemset_size;
	int freq;
	struct plain_item *item_next;
	struct trans_node *tx_next;
}itemset;


typedef struct{
	int index;
	struct eq_class *next;
}eq_class;

typedef struct{
	int threadid;
	int start_index_ip;
	int end_index_ip;
	int start_index_op;
	int end_index_op; //this value should be set by the thread for use by the main thread
	int support;
	int itemset_size;
	itemset* itemset_ip;
	itemset* itemset_op;
}arg2_list;

typedef struct{
	plain_item *rule1;
	plain_item *rule2;
	plain_item *next;
}rule_struct;

/*int match(plain_item *R1, int size1, plain_item *R2, int size2,itemset *itemset_array1, int arr1_size,itemset *itemset_array2, int arr2_size){
	plain_item *head1,*head2;
	head1=R1;
	head=R2;
	int i=0;
	int j=0;
	for(i=0;i<arr1_size;i++){

	}
}*/

#define TRANS_NUM 600000
#define NUM_THREADS 8

void *optimizer(void *args2){
	int threadid;


   	itemset *ip_itemset_array;
	itemset *op_itemset_array;

	arg2_list *mul_args2;
	int k,i,j;
	int cur_itemset_size;
	plain_item *temp_list_head=NULL;
	plain_item *temp_list_tail=NULL;
	plain_item *temp_item1, *temp_item2,*temp_node1,*temp_node2;
	int temp_list_len;
	plain_item *free_tmp_item;
	trans_node *free_tmp_trans;
	int l;
	plain_item *check_item1,*check_item2;
	int com_items;
	plain_item *f_item,*s_item;
	itemset is_temp;
	int item_end_n;
	plain_item* item_temp;
	trans_node* tx_temp;
	int freq,total_item;
	int start_index_ip,end_index_ip,start_index_op,end_index_op,support;
	trans_node *tx_head,*tx_tail,*ptr,*temp;
	plain_item *t1,*t2;
	plain_item *item_head;
	int found1=0;
	int found2=0;
	int pr_in_list=0;

	//initializing the arguments passed
	mul_args2 = (arg1_list*)args2;
	threadid = mul_args2->threadid;
	start_index_ip = mul_args2->start_index_ip;
	end_index_ip = mul_args2->end_index_ip;
	start_index_op = mul_args2->start_index_op;
	support = mul_args2->support;
	ip_itemset_array = mul_args2->itemset_ip;
	op_itemset_array = mul_args2->itemset_op;
	cur_itemset_size = mul_args2->itemset_size;


	//printf("\nthread:%d,start_index_ip:%d,end_index_ip:%d",threadid,start_index_ip,end_index_ip);
	freq=0;
	if(end_index_ip <=start_index_ip){
		mul_args2->end_index_op=start_index_ip;
		pthread_exit(NULL);
	}
	//cur_itemset_size = 3; //should 
	//creating itemsets of size greater than 2
	k=start_index_op; // used for keeping track of number of new itemsetsand also indexing into the result
	for(i=start_index_ip;i<end_index_ip;i++){
		for(j=i+1;j<end_index_ip;j++){
			//first step to find if intersection of transction lists is not null
			ptr = ip_itemset_array[i].tx_next; //pick up tx list of first itemset in array
			while(ptr!=NULL){
				temp=ip_itemset_array[j].tx_next;  //pick up head of tx list of second itemset in array
				while(temp!=NULL){
						if(temp->value==ptr->value){
							t1=ip_itemset_array[i].item_next;
							t2=ip_itemset_array[j].item_next;
							//printf("\ni:%d,j;%d,I1:%s,I2:%s,tx_value:%d,freq:%d",i,j,t1->Storecode,t2->Storecode,temp->value,freq);
							freq++;
							if(freq==1){
								trans_node* tx_temp2= (trans_node *) malloc(sizeof(trans_node));
								if(tx_temp2==NULL){
									printf("Allocation unsuccessful at line 107");
									return 1;
								}
								tx_temp2->value=temp->value;
								tx_temp2->next=NULL;
								tx_head=tx_temp2;
								tx_tail=tx_temp2;
							}
							if(freq>1){
								trans_node* tx_temp3= (trans_node *) malloc(sizeof(trans_node));
								if(tx_temp3==NULL){
									printf("Allocation unsuccessful at line 118");
									return 1;
								}
								tx_temp3->value=temp->value;
								tx_temp3->next=NULL;
								tx_tail->next=tx_temp3;
								tx_tail= tx_tail->next;
							}
							break; //an itemset cannot have two txs of same type, ex:AA(there can be only one A), so break and search for next common tx
						}
					temp =temp->next;
				}
				ptr=ptr->next;
			}
			if(freq>support){
				//creating itemset start
				temp_item1 = ip_itemset_array[i].item_next;
				temp_item2 = ip_itemset_array[j].item_next;
				temp_list_head=NULL;
				temp_list_len = 0;
				while((temp_item1!=NULL)&&(temp_item2!=NULL)){
					if(strcmp(temp_item1->Storecode,temp_item2->Storecode)!=0){
						item_head = temp_list_head;
						while(item_head!=NULL){ //remember to initialize at end
							if(strcmp(item_head->Storecode,temp_item1->Storecode)==0){
								found1=1;
								break;
							}
							item_head = item_head->next;
						}

						item_head = temp_list_head;
						while(item_head!=NULL){ //remember to initialize at end
							if(strcmp(item_head->Storecode,temp_item2->Storecode)==0){
								found2=1;
								break;
							}
							item_head = item_head->next;
						}


						if(!found1 && found2){
							plain_item *temp_node1 = (plain_item *)malloc(sizeof(plain_item));
							if(temp_node1==NULL){
								printf("Allocation unsuccessful at line 141");
								return 1;
							}
							strcpy(temp_node1->Storecode,temp_item1->Storecode);
							temp_node1->next=NULL;
							temp_list_len++;
							if(temp_list_head==NULL){
								temp_list_head = temp_node1;
								temp_list_tail = temp_node1;
							}
							else{
								temp_list_tail->next = temp_node1;
								temp_list_tail = temp_node1;
							}

						}
						else if(found1 && !found2){
							plain_item *temp_node2 = (plain_item *)malloc(sizeof(plain_item));
							if(temp_node2==NULL){
								printf("Allocation unsuccessful at line 146");
								return 1;
							}
							strcpy(temp_node2->Storecode,temp_item2->Storecode);
							temp_node2->next=NULL;
							temp_list_len++;
							if(temp_list_head==NULL){
								temp_list_head = temp_node2;
								temp_list_tail = temp_node2;
							}
							else{
								temp_list_tail->next = temp_node2;
								temp_list_tail = temp_node2;
							}
						}
						else if(!found1 && !found2){
							plain_item *temp_node1 = (plain_item *)malloc(sizeof(plain_item));
							if(temp_node1==NULL){
								printf("Allocation unsuccessful at line 141");
								return 1;
							}
							plain_item *temp_node2 = (plain_item *)malloc(sizeof(plain_item));
							if(temp_node2==NULL){
								printf("Allocation unsuccessful at line 146");
								return 1;
							}									
							//possibly add code to do arrange in correct order
							strcpy(temp_node1->Storecode,temp_item1->Storecode);
							strcpy(temp_node2->Storecode,temp_item2->Storecode);
							temp_node2->next = NULL;
							temp_node1->next = temp_node2;
							if(temp_list_head==NULL){
								temp_list_head = temp_node1;
								temp_list_tail = temp_node2;
							}
							else{
								temp_list_tail->next = temp_node1;
								temp_list_tail = temp_node2;
							}
							temp_list_len = temp_list_len + 2;
						}

					}
					else{
						plain_item *temp_node3 = (plain_item *)malloc(sizeof(plain_item));
						if(temp_node3==NULL){
							printf("Allocation unsuccessful at line 168");
							return 1;
						}
						strcpy(temp_node3->Storecode,temp_item1->Storecode);
						temp_node3->next = NULL;
						if(temp_list_head==NULL){
							temp_list_head = temp_node3;
							temp_list_tail = temp_node3;
							temp_list_len = temp_list_len + 1;
						}
						else{
							temp_list_tail->next = temp_node3;
							temp_list_tail = temp_node3;
							temp_list_len = temp_list_len + 1;
						}
					}
					temp_item1 = temp_item1->next;
					temp_item2 = temp_item2->next;
					found1=0;
					found2=0;
				}
				//create itemst end

				if(temp_list_len==cur_itemset_size){ //if greater obviously it is a candidate for the next itemset list
					//check if itemset is already present in created itemset list, not done right now
					com_items=0;
					pr_in_list=0;
					for(l=start_index_op;l<k;l++){ // check through already created list
						check_item1 = temp_list_head; //points to current itemset created
						while(check_item1!=NULL){
							check_item2 = op_itemset_array[l].item_next;
							while(check_item2!=NULL){
								if(strcmp(check_item1->Storecode,check_item2->Storecode)==0){
									com_items++;
									break;
								}
								check_item2 = check_item2->next;
							}
							check_item1 = check_item1->next;
						}
						if(com_items==cur_itemset_size){
							pr_in_list=1;
							break;
						}
					}
					if(pr_in_list){
						//this itemset is already present, so free	bot tx and item list
						free_tmp_trans = tx_head; //tx
						free_tmp_item = temp_list_head; //item
						//deleting item list
						while(temp_list_head != NULL){
							free_tmp_item = temp_list_head;
							temp_list_head = temp_list_head->next;
							free(free_tmp_item);
						}
						//deleting tx list
						while(tx_head != NULL){
							free_tmp_trans = tx_head;
							tx_head = tx_head->next;
							free(free_tmp_trans);
						}
						pr_in_list=0;
					}
					else{ //this is something we can include in the itemset list
						//printf("\n%d",k);
						op_itemset_array[k].item_next=temp_list_head; //itemset linked list
						op_itemset_array[k].tx_next=tx_head; //tx linked list
						op_itemset_array[k].freq=freq;
						op_itemset_array[k].itemset_size=3;
						freq=0;
						k++;
						temp_list_head=NULL;
						tx_head=NULL;
					}
				}
				else{//not a candidate and should be freed;
					free_tmp_trans = tx_head; //tx
					free_tmp_item = temp_list_head; //item
					//deleting item list
					while(temp_list_head != NULL){
						free_tmp_item = temp_list_head;
						temp_list_head = temp_list_head->next;
						free(free_tmp_item);
					}
					//deleting tx list
					while(tx_head != NULL){
						free_tmp_trans = tx_head;
						tx_head = tx_head->next;
						free(free_tmp_trans);
					}
				}
			}//added to list if condition satisfied
			freq=0;
		}//for loop 2
	}//for loop 1

	item_end_n=k;
	mul_args2->end_index_op = k; //for use by mainthread;
	//lexicographic sorting of itemset of size n>2
	for (i = start_index_op; i < item_end_n; i++) {
        for (j = i + 1; j < item_end_n; j++) {
            f_item = op_itemset_array[i].item_next;
            s_item = f_item->next;
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
		                is_temp=op_itemset_array[i];
		                op_itemset_array[i]=op_itemset_array[j];
		                op_itemset_array[j]=is_temp;
		            }           		
            	}
            }
        }
    }

    //printf("exiting thread%d",threadid);
	pthread_exit(NULL);	
}


void *freq_cnt(void *args) {
	transaction *trans_array;
	item *item_array;
	arg1_list *mul_args;
	long tid;
	int start,end;
	int i,j,k,m;
	int present=0;
	m=0;
	int dist_freq; // distinct frequencies
	char item_code[20];
	trans_node* tail;
	trans_node* temp;


	mul_args = (arg1_list*)args;	
	tid = mul_args->threadid;
	trans_array = mul_args->tx_list;
	item_array = mul_args->item_list;
	start=(mul_args->num_trans/NUM_THREADS)*tid;
   	end=start+(mul_args->num_trans/NUM_THREADS);
	dist_freq=start;

	//printf("\nEntering thread %d\n", tid);

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
			trans_node* temp = (trans_node*)malloc(sizeof(trans_node));
			if(temp==NULL){
				printf("Allocation unsuccessful at line 349");
				return 1;
			}
			temp->value=trans_array[i].trans_no;
			temp->next=NULL;
			item_array[dist_freq].next=temp;
			tail=(trans_node *)item_array[dist_freq].next;

			for(k=i+1;k<(end-1);k++){ //add all other items
				if(strcmp(item_code, trans_array[k].Storecode)==0){
					item_array[dist_freq].freq++;
					trans_node* temp = (trans_node*)malloc(sizeof(trans_node));
					if(temp==NULL){
						printf("Allocation unsuccessful at line 362");
						return 1;
					}
					temp->value=trans_array[k].trans_no;
					temp->next=NULL;
					tail->next=(trans_node *)temp;
					tail=(trans_node *)tail->next;
				}
			}
			dist_freq++;
		}
		else{
			present=0;
			continue;	
		}
	}
	strcpy(item_array[dist_freq].Storecode,"end_of_item_list");

	//printf("\nLeaving thread %d\n", tid);
	pthread_exit(NULL);
}


int main(void) {
	int rc,j,k,l,m, present;
	int support;
    char *buf;
	char item_num[20];
    pthread_t threads[NUM_THREADS];

   	//itemset *ip_itemset_array;
	itemset *op_itemset_array;

	itemset *ip_itemset_array_1;
	itemset *ip_itemset_array_2;
	itemset *op_itemset_array_3;
	itemset *ip_itemset_array_3;
	itemset *op_itemset_array_4;
	itemset *ip_itemset_array_4;
	itemset *op_itemset_array_5;
	itemset *ip_itemset_array_5;
	itemset *op_itemset_array_6;
	itemset *ip_itemset_array_6;
	itemset *op_itemset_array_7;
	itemset *ip_itemset_array_7;
	itemset *op_itemset_array_8;	

	transaction* trans_array = (transaction*)malloc(sizeof(transaction)*TRANS_NUM);
	if(trans_array==NULL){
		printf("Allocation unsuccessful at line 396");
		return 1;
	}
	item* item_array = (item*)malloc(sizeof(item)*TRANS_NUM);
	if(item_array==NULL){
		printf("Allocation unsuccessful at line 401");
		return 1;
	}
	item* item_frequency = (item*)malloc(sizeof(item)*TRANS_NUM);
	if(item_frequency==NULL){
		printf("Allocation unsuccessful at line 406");
		return 1;
	}
	item* item_frequency_min_sup = (item*)malloc(sizeof(item)*TRANS_NUM);
	if(item_frequency_min_sup==NULL){
		printf("Allocation unsuccessful at line 411");
		return 1;
	}

	memset(item_frequency_min_sup,0,sizeof(item)*TRANS_NUM);
	memset(item_frequency,0,sizeof(item)*TRANS_NUM);
	memset(item_array,0,sizeof(item)*TRANS_NUM);
	memset(trans_array,0,sizeof(transaction)*TRANS_NUM);

	arg1_list *args[NUM_THREADS]; // for passing to the threads
	arg2_list *args2[NUM_THREADS]; // for passing to the threads
	trans_node *temp,*temp1,*tail;
	int item_start,item_end_1,item_end_2,item_end_3,item_end_4,item_end_5,item_end_6,item_end_7,item_end_8,item_end_9,item_num_freq;
	item topo_temp;
	int itemset_no;
	//itemset* itemset_array_2;
	int freq=0;
	item first;
	trans_node* ptr;
	int itemset_size,iter;
	item first_ptr;
	int match_found;
	plain_item *f_item,*s_item;
	plain_item *item_temp,*item_temp1;
	itemset is_temp;
	plain_item *holder1;
	plain_item *holder2;
	int first_value;
	trans_node *tx_head,*tx_tail,*tx_temp;
	plain_item *prev;
	plain_item *cur;
	eq_class *class_head,*class_tail,*class_temp,*node;
	int confidence;

	double exec_time = 0.0;


	int r,total_op_size,count,start_index,final_index,cutoff,count2,range,op_array_index;

	temp = (trans_node *)malloc(sizeof(trans_node)); //seg_fault
	if(temp==NULL){
		printf("Allocation unsuccessful at line 443");
		return 1;
	}
	temp1 = (trans_node *)malloc(sizeof(trans_node)); //seg_fault
	if(temp1==NULL){
		printf("Allocation unsuccessful at line 448");
		return 1;
	}


	//printf("\nEnter minimum support count:");
	//scanf("%d",&support);
	support=3;

	clock_t start = clock();


    FILE *fp = fopen("ex2.csv", "r"); //"Online_Retail.csv  ex2 ex_5000
    //FILE *fp = fopen("Online_Retail.csv", "r"); //"Online_Retail.csv
    if (!fp) {
        printf("Can't open file\n");
        return 0;
    }
	size_t len=0;
	int row_count=0;
    int i = 0;
    int field_count = 0;

//    args=(arg1_list *)malloc(sizeof(arg1_list));
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
	fclose(fp);

	for(i=0;i<NUM_THREADS;i++){ //NUM_THREADS
		args[i]=(arg1_list *)malloc(sizeof(arg1_list));
		if(args[i]==NULL){
			printf("Allocation unsuccessful at line 498");
			return 1;
		}
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

	free(trans_array);

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


	printf("\nTotal distinct words:%d",item_end_1);


//Stage2
//creating itemsets of size 2 and more
	if(itemset_no==0){
		printf("\n maximum itemset size is %d", itemset_no);
		return;
	}


	ip_itemset_array_2 = (itemset *)malloc((sizeof(itemset))*(item_end_1*item_end_1));
	if(ip_itemset_array_2==NULL){
		printf("Allocation unsuccessful at line 643");
		return 1;
	}
	//int tttt = sizeof((item_end_1*item_end_1)/2);
	//printf("%d\n", tttt);


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
							if(freq==1){
								trans_node* tx_temp2= (trans_node *) malloc(sizeof(trans_node));
								if(tx_temp2==NULL){
									printf("Allocation unsuccessful at line 664");
									return 1;
								}
								tx_temp2->value=temp->value;
								tx_temp2->next=NULL;
								tx_head=tx_temp2;
								tx_tail=tx_temp2;
								//printf("\n");
								//printf("\n Debug_info");
								//printf("\nitemset:%d tx_head:%llx tx_head_nxt:%llx\t",k,tx_head,tx_head->next);
							}
							if(freq>1){
								trans_node* tx_temp3= (trans_node *) malloc(sizeof(trans_node));
								if(tx_temp3==NULL){
									printf("Allocation unsuccessful at line 678");
									return 1;
								}
								tx_temp3->value=temp->value;
								tx_temp3->next=NULL;
								tx_tail->next=tx_temp3;
								tx_tail= tx_tail->next;
								//if(freq==2){
								//printf("\nitemset:%d tx_head:%llx tx_head_nxt:%llx\t",k,tx_head,tx_head->next);
								//}
							}
							break;
						}
					temp =temp->next;
				}
				ptr=ptr->next;
			}
			if(freq>support){
				plain_item* holder1 = (plain_item *)malloc(sizeof(plain_item));
				if(holder1==NULL){
					printf("Allocation unsuccessful at line 698");
					return 1;
				}

				plain_item* holder2 = (plain_item *)malloc(sizeof(plain_item));
				if(holder2==NULL){
					printf("Allocation unsuccessful at line 704");
					return 1;
				}
				strcpy(holder1->Storecode,item_frequency_min_sup[i].Storecode);
				strcpy(holder2->Storecode,item_frequency_min_sup[j].Storecode);
				holder1->next=holder2;
				holder2->next=NULL;					
				ip_itemset_array_2[k].item_next=holder1;
				ip_itemset_array_2[k].tx_next=tx_head;
				ip_itemset_array_2[k].freq=freq;
				ip_itemset_array_2[k].itemset_size=2;
				freq=0; //not necessary, kind of redundant
				k++;
			}
			freq=0;
		}
	}

	item_end_2=k; // no of elements in the itemset of size 2
	printf("\nNo of itemsets of size 2:%d\n",item_end_2);


	//lexicographic sorting of itemset of size 2
	for (i = 0; i < item_end_2; i++) {
        for (j = i + 1; j < item_end_2; j++) {
            f_item = ip_itemset_array_2[i].item_next;
            s_item = f_item->next;
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
		                is_temp=ip_itemset_array_2[i];
		                ip_itemset_array_2[i]=ip_itemset_array_2[j];
		                ip_itemset_array_2[j]=is_temp;
		            }           		
            	}
            }
        }
    }


    //ITEMSET2 itemset_array_2
    /*printf("\n lexigographically sorted itemset2");
	for(i=0;i<item_end_2;i++){
		tx_temp=ip_itemset_array_2[i].tx_next;
		item_temp=ip_itemset_array_2[i].item_next;
		item_temp1=item_temp->next;
		//printf("\nItem1:%s, Item2:%s, tx list:\n",holder1->Storecode,holder2->Storecode);
		printf("\nItem1:%s, Item2:%s, Freq:%d, tx list:\n",item_temp->Storecode,item_temp1->Storecode,ip_itemset_array_2[i].freq);
		while(tx_temp!=NULL){
			printf("%d\t",tx_temp->value);
			tx_temp=tx_temp->next;
		}
	}*/



////////////////////////////////////////////////////////////////////////////
	//ITEMSET3 creation start

//	op_itemset_array = (itemset *)malloc((sizeof(itemset))*item_end_2);
	op_itemset_array_3 = (itemset *)malloc((sizeof(itemset))*500000000);
   	if(op_itemset_array_3==NULL){
		printf("Allocation unsuccessful at line 797");
		return 1;
	}


	//Identify equivalence classes
	item_temp=ip_itemset_array_2[0].item_next;
	prev = (plain_item *)malloc(sizeof(plain_item));
	cur = (plain_item *)malloc(sizeof(plain_item));
	strcpy(prev->Storecode,item_temp->Storecode);
	prev->next=NULL;
	cur->next=NULL;
	node= (eq_class *)malloc(sizeof(eq_class));
	node->index=0;
	node->next=NULL;
	class_head=node;
	class_tail=node;
	count=0;
	for(i=1;i<item_end_2;i++){
		item_temp=ip_itemset_array_2[i].item_next;
		strcpy(cur->Storecode,item_temp->Storecode);
		if(strcmp(cur,prev->Storecode)==0){
			continue;
		}
		else{
			eq_class *node= (eq_class *)malloc(sizeof(eq_class));
			node->index=i;
			node->next=NULL;
			class_tail->next=node;
			class_tail=class_tail->next;
			strcpy(prev->Storecode,cur->Storecode);
			count++;
		}
	}

	//printf("\n count%d",count);

	//printing index of equivalence class
	start_index=class_head->index;
	final_index=0;
	cutoff=ceil(count/NUM_THREADS);
	if((count%NUM_THREADS)!=0){
		cutoff++;
	}
	//printf("\n cutoff%d",cutoff);
	count2=0;
	//printf("\nEquivalence class");
	class_temp=class_head;

//Stage3
	//int start_index;
	op_array_index=0;// the first index
	range;
	start_index = class_temp->index;
	//int end_index;
	for(i=0;i<NUM_THREADS;i++){ //NUM_THREADS
		count2=0;
		//start_index = class_temp->index;
		args2[i]=(arg2_list *)malloc(sizeof(arg2_list));
	   	if(args2[i]==NULL){
			printf("Allocation unsuccessful at line 805");
			return 1;
		}
		
		/*while(class_temp!=NULL){
			//printf("\n%d",class_temp->index);
			class_temp=class_temp->next;
			count2=count2+1;
			if(count2==cutoff){
				final_index=(class_temp->index)-1;
				printf("\nthread:%d start_index:%d final_index:%d",i,start_index,final_index);
				break;
			}
		}*/

		while(class_temp!=NULL){
			//printf("\n%d",class_temp->index);
			
			count2=count2+1;
			if(count2==cutoff){
				final_index=(class_temp->index)-1;
				class_temp=class_temp->next;
				//printf("\nthread:%d start_index:%d final_index:%d",i,start_index,final_index);
				break;
			}
			class_temp=class_temp->next;
		}


		args2[i]->threadid=i;
		args2[i]->start_index_ip=start_index;
		args2[i]->end_index_ip=final_index;
		args2[i]->itemset_size=3;
		args2[i]->support=support;
		args2[i]->start_index_op=op_array_index;
		range=final_index-start_index;
		//printf("\n thread:%d range:%d op_array_index%d",i,range,op_array_index);
		op_array_index +=100000000/NUM_THREADS;
		//op_array_index += ((range)*(range))/2;
		args2[i]->itemset_ip = ip_itemset_array_2; //itemset_array_2
		args2[i]->itemset_op = op_itemset_array_3;
		rc=pthread_create(&threads[i], NULL, optimizer, args2[i] );
		if (rc) {
			printf("Error:unable to create thread, %d\n", rc);
			exit(-1);
		}
		start_index =final_index+1;
	}

	r=0;
	total_op_size=0;
	for(i=0;i<NUM_THREADS;i++){
		pthread_join(threads[i], NULL);
		//printf("\nthread%d start_index:%d end_index:%d",i,args2[i]->start_index_op,args2[i]->end_index_op);
		total_op_size+=(args2[i]->end_index_op-args2[i]->start_index_op);
	}
	

	ip_itemset_array_3 = (itemset *)malloc((sizeof(itemset))*(item_end_2*item_end_2));
	for(i=0;i<NUM_THREADS;i++){
		for(j=args2[i]->start_index_op;j<args2[i]->end_index_op;j++){
			ip_itemset_array_3[r]=op_itemset_array_3[j];
			r++;
		}
	}
	item_end_3=r;
	printf("\nNo of itemsets of size 2:%d\n",item_end_3);


    /*printf("\nlexigographically sorted itemset3:\n");
	for(i=0;i<r;i++){
		tx_temp=ip_itemset_array_3[i].tx_next;
		item_temp=ip_itemset_array_3[i].item_next;
		//item_temp1=item_temp->next;
		printf("\n");
		while(item_temp!=NULL){
			printf("Item:%s\t",item_temp->Storecode);
			item_temp = item_temp->next;
		}
		printf("freq:%d\t",ip_itemset_array_3[i].freq);
		printf("tx_list:\n");
		while(tx_temp!=NULL){
			printf("%d\t",tx_temp->value);
			tx_temp=tx_temp->next;
		}
	}*/
//////////////////////////////////////////////////////////////////////





////comment from this onwards

//printf("\nenter confidence in percentage:");
//scanf("%d",&confidence);
confidence=80;
rule_struct* rule_array = (rule_struct *)malloc((sizeof(rule_struct))*1000000);
int z=0;

//Association Rules Generation
	plain_item *rule_temp=malloc(sizeof(plain_item)*(3));
	plain_item *rule_temp1=malloc(sizeof(plain_item));
	plain_item *list1_head,*list2_head,*list1_temp,*list2_temp;
	plain_item *temp_rule_list,*rule_head,*rule_tail,*r1_temp,*r2_temp;
	int match_count,exit_count;
	int R1_freq,R2_freq;
	int q;
	int rule_fnd,fnd_cnt;
	int found1,found2;

	//itemset of size 1
	for(i=0;i<item_end_3;i++){
		rule_temp1=op_itemset_array_3[i].item_next;
		j=0;
		while(rule_temp1!=NULL){
			strcpy(rule_temp[j].Storecode,rule_temp1->Storecode);
			j++;
			rule_temp1=rule_temp1->next;
		}
		for(l=0;l<2;l++){ //1,(n-1)
			//1st rule
			for(k=0;k<item_end_1;k++){//searching through itemset array for match
				if(strcmp(rule_temp[l].Storecode,item_frequency_min_sup[k].Storecode)==0){
					found1=1;
					break;
				}
			}
			R1_freq=item_frequency_min_sup[k].freq;

			//2nd rule
			q=0;
			rule_head=NULL;//points to the second rule
			while(q<3){ //itemset_size
				if(q==l){
					q++;
					continue;
				}
				temp_rule_list = (plain_item *)(malloc(sizeof(plain_item)));
				strcpy(temp_rule_list->Storecode,rule_temp[q].Storecode);
				temp_rule_list->next = NULL;
				q++;
				if(rule_head==NULL){
					rule_head=temp_rule_list;
					rule_tail=temp_rule_list;
				}
				else{
					rule_tail->next=temp_rule_list;
					rule_tail=rule_tail->next;
				}
			}//candidate 2nd rule created
			match_count=0;
			rule_fnd=0;
			for(r=0;r<item_end_2;r++){//search for the rule in itemset array for match
				r1_temp=rule_head; //rule2 under consideration
				fnd_cnt=0;
				while(r1_temp!=NULL){
					r2_temp=ip_itemset_array_2[r].item_next;
					while(r2_temp!=NULL){
						if(strcmp(r1_temp->Storecode,r2_temp->Storecode)==0){
							fnd_cnt++;
							break;
						}
						r2_temp=r2_temp->next;
					}
					r1_temp=r1_temp->next;
				}
				if(fnd_cnt==2){
					break;
				}
			}
			R2_freq=item_frequency_min_sup[k].freq;
			if(((R1_freq*100)/R2_freq)>confidence){
				plain_item *first_rule=(plain_item *)malloc(sizeof(plain_item));
				strcpy(first_rule->Storecode,item_frequency_min_sup[k].Storecode);
				first_rule->next=NULL;
				rule_array[z].rule1=first_rule;
				rule_array[z].rule2=rule_head;
				z++;
			}

		}
	}
	int num_thread=NUM_THREADS;
	clock_t end = clock();
	exec_time += (double)(end - start) / CLOCKS_PER_SEC;
	printf("Exection Time: %f seconds for %d threads", exec_time,num_thread);
	printf("end");

/*
	for(i=0;i<z;i++){
		item_temp=rule_array[i].rule1;
		item_temp1=rule_array[i].rule2;
		printf("\n first rule:%s, second rule:%s->",item_temp->Storecode,item_temp1->Storecode);
		item_temp1=item_temp1->next;
		printf("%s",item_temp1->Storecode);
	}
*/

	pthread_exit(NULL);
	return 0;
}
