#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>

#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <numeric>

//#include "lock.h"

using namespace std;

typedef struct trans_node {
	int value;
} TransNode;

typedef struct {
	int trans_no;
    int item_size;
    int item_code[1024];
} Transaction;

typedef struct {
	int item_no;
    int freq;
    int trans_array_size;
    int trans_array[256];
} Item;

typedef struct {
    int freq;
    int item_set_size;
	int item_set_code[256];
    int trans_array_size;
    int trans_array[256];

    /* the indices of previous sets */
    int set1_index;
    int set2_index;

} ItemSet;


#define TRANS_NUM 600000
#define NUM_THREADS 16


__global__ 
void item_freq_count(int num_trans, Transaction *transArray, Item* itemArray)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = gridDim.x*blockDim.x;
    int i = tid;
    while ( i < num_trans) {
        int item_size = transArray[i].item_size;                 
        for (int j = 0; j < item_size; j++) {
            int item_code = transArray[i].item_code[j];
            //itemArray[item_code].freq++;
            atomicAdd(&(itemArray[item_code].freq), 1);
            /* push the transaction to the item struct */
            int _idx = atomicAdd(&(itemArray[item_code].trans_array_size), 1);
            itemArray[item_code].trans_array[_idx] = i;
        }
        i += num_threads;
    }
}

__global__
void select_with_min_support(int num_items, Item* itemArray, int min_support, ItemSet* itemsetArray, int* globalIdx)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    int i = tid;
    while (i < num_items) { 
        if (itemArray[i].freq >= min_support) {
            /* get a place in itemsetArray */
            int _idx = atomicAdd(globalIdx, 1);
            itemsetArray[_idx].freq = itemArray[i].freq;
            itemsetArray[_idx].item_set_size = 1;
            itemsetArray[_idx].item_set_code[0] = itemArray[i].item_no;
            itemsetArray[_idx].trans_array_size = itemArray[i].trans_array_size;
            memcpy(itemsetArray[_idx].trans_array, itemArray[i].trans_array, itemArray[i].trans_array_size*sizeof(int));
        }
        i += num_threads;
    }
}


__device__
bool alreadyHasTrans(ItemSet* _item_set, int trans_no)
{
    bool has = false;
    for (int i = 0; i < _item_set->trans_array_size; i++ ) {
        if (_item_set->trans_array[i] == trans_no) {
            has = true;
        }
    }
    return has;
}

/* search for transactions in the previous itemset, updating the transaction records
    and returning the count */
__device__
int find_support_count_for_itemset(ItemSet* candidate_itemset, ItemSet* checked_itemset, Transaction* trans_array)
{
    int count = 0;
    for (int i = 0; i < checked_itemset->trans_array_size; i++) {
        int trans_idx = checked_itemset->trans_array[i];
        Transaction* trans = &(trans_array[trans_idx]);
        bool itemset_found = true;
        int trans_no = -1;
        for (int j = 0; j < candidate_itemset->item_set_size; j++) {
            int target_item_code = candidate_itemset->item_set_code[j];
            bool single_item_found = false;
            for (int k = 0; k < trans->item_size; k++) {
                if (target_item_code == trans->item_code[k] && 
                    !alreadyHasTrans(candidate_itemset, trans_idx)) {
                    single_item_found = true; 
                    trans_no = trans_idx;
                    break;
                }
            }
            itemset_found &= single_item_found;
        }
        if (itemset_found) {
            candidate_itemset->trans_array[candidate_itemset->trans_array_size++] = trans_no;
            count++;
        }
    }
    return count;
}

__global__
void find_support_count(int* candidateSetSize, ItemSet* candidateSet, int* globalIdx, ItemSet* currSet, Transaction* trans_array)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    int i = tid;
    
    while (i < *candidateSetSize) {
        int set1_idx = candidateSet[i].set1_index;
        int set2_idx = candidateSet[i].set2_index;
        int count1 = find_support_count_for_itemset(&(candidateSet[i]), &(currSet[set1_idx]), trans_array);
        int count2 = find_support_count_for_itemset(&(candidateSet[i]), &(currSet[set2_idx]), trans_array);
        
        int _global_idx = atomicAdd(globalIdx, 1);
        candidateSet[_global_idx].freq = count1 + count2;

        i += num_threads;
    }

}

int itemsetComp(const void* a, const void* b)
{
    ItemSet* set_a = (ItemSet*)(a);
    ItemSet* set_b = (ItemSet*)(b);
    int size = set_a->item_set_size;
    for (int i = 0; i < size; i++) {
        if (set_a->item_set_code[i] > set_b->item_set_code[i]) {
            return 1;
        }
        else if (set_a->item_set_code[i] < set_b->item_set_code[i]) {
            return -1;
        }
    }
    return 0;
}


int find_last_eq_class_item(int array_size, ItemSet* itemset_array, int base_pos, int start_pos, int cardinality)
{
    ItemSet* base_item_set = &(itemset_array[base_pos]);
    int last_pos = -1;
    
    if (cardinality < 2) {
        return -1;
    }

    for (int i = start_pos; i < array_size; i++) {
        ItemSet* check_item_set = &(itemset_array[i]);
        for (int j = 0; j < cardinality-1; j++) {
            if (base_item_set->item_set_code[j] != check_item_set->item_set_code[j]) {
                break; 
            }
        }
        last_pos = i;
    }

    return last_pos;
}

void* genNextItemSetArray(int itemset_array_size, ItemSet* curr_itemset_array, int nextCardinality, int* nextSize)
{
    int _arr_size = itemset_array_size;
    int new_idx = 0;
    if (itemset_array_size <= 0) {
        return NULL;
    }
    
    assert(nextCardinality == curr_itemset_array[0].item_set_size);
    
    ItemSet* next_set = NULL;
    
    if (nextCardinality == 2) {
        int next_size = (_arr_size*(_arr_size-1)) / 2;
        next_set = (ItemSet*)malloc(next_size*sizeof(ItemSet));
        memset(next_set, 0, next_size*sizeof(ItemSet));
        for (int i = 0; i < _arr_size-1; i++) {
            for (int j = i+1; j < _arr_size; j++) {
                /* set up new itemset */
                next_set[new_idx].item_set_size = nextCardinality;
                next_set[new_idx].item_set_code[0] = curr_itemset_array[i].item_set_code[0];
                next_set[new_idx].item_set_code[1] = curr_itemset_array[j].item_set_code[0];
                
                /* store the indices */
                next_set[new_idx].set1_index = i;
                next_set[new_idx].set2_index = j;

                new_idx++;
            }
        }
        *nextSize = next_size;
    }
    else {
        int i = 0;
        vector< pair<int,int> > ranges_vec;
        while (i < itemset_array_size) {
            int j = find_last_eq_class_item(itemset_array_size, curr_itemset_array, i, i+1, nextCardinality-1);
            ranges_vec.push_back(make_pair(i,j));
            i = j+1; 
        }
       
        auto pairSum = [](vector< pair<int,int> >& _vec) {
            int sum = 0;
            for (int i = 0; i < _vec.size(); i++) {
                sum += (_vec[i].second-_vec[i].first+1);
            }
            return sum;
        };
        /* allocate next level item set memory */ 
        int next_size = pairSum(ranges_vec);
        next_set = (ItemSet*)malloc(next_size*sizeof(ItemSet));
        memset(next_set, 0, next_size*sizeof(ItemSet));
        for (auto range : ranges_vec) {
            /* the priori nextCardinality-2 items should be the same */
            for (int start_pos = range.first; start_pos <= range.second-1; start_pos++) {
                for (int end_pos = start_pos+1; end_pos <= range.second; end_pos++) {
                    /* set up new itemset */
                    next_set[new_idx].item_set_size = nextCardinality;
                    
                    memcpy(next_set[new_idx].item_set_code,
                           curr_itemset_array[start_pos].item_set_code,
                           curr_itemset_array[start_pos].item_set_size*sizeof(int));
                    
                    next_set[new_idx].item_set_code[nextCardinality-1] = curr_itemset_array[end_pos].item_set_code[nextCardinality-2];
                    
                    /* store the indices */
                    next_set[new_idx].set1_index = start_pos;
                    next_set[new_idx].set2_index = end_pos;
    
                    new_idx++; 
                }
            }
        }
        *nextSize = next_size;
    }

    return (void*)next_set;
}

int main(void) 
{
    fstream fs;
    string line;
    unordered_map<string, int> item_code_map;
    unordered_map<int, int> transaction_map;

    int trans_count = 0;    /* number of transactions */
    int item_count = 0;     /* number of unique items */
    int min_support = 5;    /* mininum supoort of items */

    Transaction *transArray = (Transaction*)malloc(TRANS_NUM*sizeof(Transaction));
    memset(transArray, 0, TRANS_NUM*sizeof(Transaction));

    /* read from the file */
    fs.open("ex_data.csv", ios::in);
    while (getline(fs, line)) {
        /* get transaction number */
        ssize_t pos = line.find(",");
        int trans_no = atoi(line.substr(0, pos).c_str());
        ssize_t pos2 = line.find(",", pos+1);
        string item = line.substr(pos+1, pos2-pos-1);

        /* find item number */
        if (item_code_map.find(item) == item_code_map.end()) {
            item_code_map[item] = item_count++;
        }
        /* find transaction number */
        if (transaction_map.find(trans_no) == transaction_map.end()) {
            transArray[trans_count].trans_no = trans_count;
            transArray[trans_count].item_code[transArray[trans_count].item_size++] = item_code_map[item];
            transaction_map[trans_no] = trans_count;
            trans_count++;
        }
        else {
            int _idx = transaction_map[trans_no]; 
            transArray[_idx].item_code[transArray[_idx].item_size++] = item_code_map[item];
        }
    }
    fs.close();
    
    Item *itemArray = (Item*)malloc(item_count*sizeof(Item));
    memset(itemArray, 0, item_count*sizeof(Item));
    for (int i = 0; i < item_count; i++) {
        itemArray[i].item_no = i;
    }
    
    /* request cuda memory */
    Transaction *dev_transArray = NULL;
    cudaMalloc(&dev_transArray, TRANS_NUM*sizeof(Transaction));
    cudaMemcpy(dev_transArray, transArray, TRANS_NUM*sizeof(Transaction), cudaMemcpyHostToDevice);
    
    Item *dev_itemArray = NULL;
    cudaMalloc(&dev_itemArray, item_count*sizeof(Item));
    cudaMemcpy(dev_itemArray, itemArray, item_count*sizeof(Item), cudaMemcpyHostToDevice);
    
    /* calculate single item frequency */
    dim3 gridSize(256);
    dim3 blockSize(16);
    item_freq_count<<<gridSize, blockSize>>>(trans_count, dev_transArray, dev_itemArray);

    /* copy the results back to host */
    cudaMemcpy(itemArray, dev_itemArray, item_count*sizeof(Item), cudaMemcpyDeviceToHost);

    /* start to prune */
    int globalIdx = 0;
    int *dev_globalIdx = NULL;
    cudaMalloc(&dev_globalIdx, sizeof(int));
    cudaMemcpy(dev_globalIdx, &globalIdx, sizeof(int), cudaMemcpyHostToDevice);

    ItemSet *itemsetArray = (ItemSet*)malloc(item_count*sizeof(ItemSet));
    memset(itemsetArray, 0, item_count*sizeof(ItemSet));
    
    ItemSet *dev_itemsetArray = NULL;
    cudaMalloc(&dev_itemsetArray, item_count*sizeof(ItemSet));
    cudaMemcpy(dev_itemsetArray, itemsetArray, item_count*sizeof(ItemSet), cudaMemcpyHostToDevice);
    
    /* kernel doing selection for single item with minimum support */
    select_with_min_support<<<gridSize, blockSize>>>(item_count, dev_itemArray, min_support, dev_itemsetArray, dev_globalIdx);

    cudaMemcpy(itemsetArray, dev_itemsetArray, item_count*sizeof(ItemSet), cudaMemcpyDeviceToHost);
    cudaMemcpy(&globalIdx, dev_globalIdx, sizeof(int), cudaMemcpyDeviceToHost);
    
    /* Now we get the transposed database that every item set with size 1 has a corresponding list of transactions */
    /* Generate itemset with size 2 */
    
    int cardinality = 2;
    int currSetSize = item_count;
    int candidateSetSize = 0;
    int *dev_candidateSetSize = NULL;
    ItemSet* currSet = itemsetArray;
    ItemSet* dev_currSet = NULL;
    ItemSet* candidateSet = NULL;
    ItemSet* dev_candidateSet = NULL;

    cudaMalloc(&dev_candidateSetSize, sizeof(int));

    while (true) {
        candidateSet = (ItemSet*)genNextItemSetArray(currSetSize, currSet, cardinality, &candidateSetSize);
        if (candidateSetSize == 0) {
            break;
        }
        assert(candidateSet != NULL);          
        
        /* allocate GPU kernel memory */
        cudaMemcpy(dev_candidateSetSize, &candidateSetSize, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_globalIdx, &globalIdx, sizeof(int), cudaMemcpyHostToDevice);
        //cudaMalloc(&dev_currSet, currSetSize*sizeof(ItemSet));
        //cudaMemcpy(dev_currSet, currSet, currSetSize*sizeof(ItemSet), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_candidateSet, candidateSetSize*sizeof(ItemSet));
        cudaMemcpy(dev_candidateSet, candidateSet, candidateSetSize*sizeof(ItemSet), cudaMemcpyHostToDevice);        
        
        /* launch the kernel */
        find_support_count<<<gridSize, blockSize>>>(dev_candidateSetSize, 
                                                     dev_candidateSet, 
                                                     dev_globalIdx, 
                                                     dev_currSet, 
                                                     dev_transArray);

        /* copy the result back */
        cudaMemcpy(candidateSet, dev_candidateSet, candidateSetSize*sizeof(ItemSet), cudaMemcpyDeviceToHost);

        /* update the parameters and free previously used memory */
        free(currSet);
        cudaFree(dev_currSet);
        cardinality++;
        currSet = candidateSet;
        currSetSize = candidateSetSize;
        dev_currSet = dev_candidateSet; 
        globalIdx = 0;
    }
    
    /* Finally Generate association rules */

    return 0;
}
