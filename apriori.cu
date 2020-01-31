#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <unistd.h>

#include <cuda_profiler_api.h>

#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <numeric>
#include <functional>
#include <set>
#include <chrono>

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
    int trans_array[128];
} Item;

typedef struct {
    int freq;
    int item_set_size;
	int item_set_code[16];
    int trans_array_size;
    int trans_array[16];

    /* the indices of previous sets */
    int set1_index;
    int set2_index;
    
    bool pruned;
} ItemSet;

typedef struct {
    int freq;
    int count;
    int *item_code_array;
} SupportCount;

#define TRANS_NUM 1000
#define ITEM_NUM 2000
#define NUM_THREADS 1
#define BLOCK_SIZE 1


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
            break;
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
    //printf("checked item set trans size %d\n", checked_itemset->trans_array_size);
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
void find_support_count(int candidateSetSize, ItemSet* candidateSet, int* globalIdx, ItemSet* currSet, Transaction* trans_array, int min_support)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    int i = tid;
    
    while (i < (candidateSetSize)) {
        int set1_idx = candidateSet[i].set1_index;
        int set2_idx = candidateSet[i].set2_index;
        int count1 = find_support_count_for_itemset(&(candidateSet[i]), &(currSet[set1_idx]), trans_array);
        int count2 = find_support_count_for_itemset(&(candidateSet[i]), &(currSet[set2_idx]), trans_array);
        
        candidateSet[i].freq = count1 + count2;
        /* check with minimum spport */ 
        if ((count1 + count2) >= min_support) {
            //int _global_idx = atomicAdd(globalIdx, 1);
            candidateSet[i].pruned = false;
        }
        else {
            candidateSet[i].pruned = true;
            //candidateSet[i].freq = -1;
        }
        i += num_threads;
    }
    
    /* block-level barrier */
    //__syncthreads();
}

int itemcodeComp(const void* a, const void* b)
{
    return (*(int*)a - *(int*)b);
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

bool hasTheItemSet(std::set<ItemSet*>& itemsets_set, ItemSet* checked_set) 
{
    for (auto it = itemsets_set.begin(); it != itemsets_set.end(); it++) {
        if ((*it)->item_set_size != checked_set->item_set_size) continue;
        if (memcmp((*it)->item_set_code, checked_set->item_set_code, checked_set->item_set_size*sizeof(int)) == 0) {
            return true; 
        }
    }
    return false;
}

int find_last_eq_class_item(int array_size, ItemSet* itemset_array, int base_pos, int start_pos, int cardinality)
{
    ItemSet* base_item_set = &(itemset_array[base_pos]);
    int last_pos = base_pos;
    
    if (cardinality < 2) {
        return -1;
    }

    for (int i = start_pos; i < array_size; i++) {
        ItemSet* check_item_set = &(itemset_array[i]);
        for (int j = 0; j < cardinality-1; j++) {
            if (base_item_set->item_set_code[j] != check_item_set->item_set_code[j]) {
                goto last_pos_ret; 
            }
        }
        last_pos = i;
    }

last_pos_ret:
    return last_pos;
}

void* genNextItemSetArray(int itemset_array_size, ItemSet* curr_itemset_array, int nextCardinality, int* nextSize)
{
    int _arr_size = itemset_array_size;
    int new_idx = 0;
    if (itemset_array_size <= 0) {
        return NULL;
    }
    
    assert(nextCardinality-1 == curr_itemset_array[0].item_set_size);
    
    ItemSet* next_set = NULL;
    
    if (nextCardinality == 2) {
        int next_size = (_arr_size*(_arr_size-1)) / 2;
        next_set = (ItemSet*)malloc(next_size*sizeof(ItemSet));
        assert(next_set != NULL);
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
            if ( (j != -1) && (i != j) ) {
                ranges_vec.push_back(make_pair(i,j));
            }
            i = j+1; 
        }
       
        auto pairSum = [](vector< pair<int,int> >& _vec) {
            int sum = 0;
            for (int i = 0; i < _vec.size(); i++) {
                int _size = (_vec[i].second-_vec[i].first+1);
                sum += (_size*(_size-1)/2);
            }
            return sum;
        };
        /* allocate next level item set memory */ 
        int next_size = pairSum(ranges_vec);
        if (next_size <= 0 || next_size > 1024*1024) {
            return NULL;
        }

        set<ItemSet*> itemsets_set;
        next_set = (ItemSet*)malloc((size_t)next_size*(size_t)sizeof(ItemSet));
        assert(next_set != NULL);
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
                    
                    /*
                    if (hasTheItemSet(itemsets_set, &(next_set[new_idx]))) {
                        next_size--;
                        continue;
                    }
                    itemsets_set.insert(&(next_set[new_idx]));
                    */

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

int main(int argc, char *argv[]) 
{
    fstream fs;
    string line;
    unordered_map<string, int> item_code_map;
    unordered_map<int, int> transaction_map;
    vector<SupportCount> support_count_vec;

    int trans_count = 0;    /* number of transactions */
    int item_count = 0;     /* number of unique items */
    int min_support = 6;    /* mininum supoort of items */
    
    /* profiling */
    size_t memory_use = 0, max_memory_use = 0;
    size_t dev_memory_use = 0, max_dev_memory_use = 0;

    /* args */
    int opt;
    int num_threads = 0;
    int block_size = 0;
    int item_num = ITEM_NUM;
    const char *optstr = "n:b:i:";
    while ((opt = getopt(argc, argv, optstr)) != -1) {
        switch (opt) {
            case 'n':
                num_threads = atoi(optarg);
                break;
            case 'b':
                block_size = atoi(optarg);
                break;
            case 'i':
                item_num = atoi(optarg);
                break;
        }
    }
    printf("num threads %d, block size %d\n", num_threads, block_size); 

    Transaction *transArray = (Transaction*)malloc(TRANS_NUM*sizeof(Transaction));
    memset(transArray, 0, TRANS_NUM*sizeof(Transaction));
    memory_use += TRANS_NUM*sizeof(Transaction);

    /* read from the file */
    //fs.open("test.csv", ios::in);
    fs.open("data.csv", ios::in);
    while (getline(fs, line)) {
        if (line.size() == 0) continue;
        /* get transaction number */
        ssize_t pos = line.find(",");
        int trans_no = atoi(line.substr(0, pos).c_str());
        ssize_t pos2 = line.find(",", pos+1);
        string item = line.substr(pos+1, pos2-pos-1);
        
        /* find item number */
        if (item_code_map.find(item) == item_code_map.end()) {
            item_code_map[item] = item_count++;
            //printf("Item Count :%d -> %s\n", item_count, item.c_str());
        }
        /* find transaction number */
        if (transaction_map.find(trans_no) == transaction_map.end()) {
            transArray[trans_count].trans_no = trans_count;
            transArray[trans_count].item_code[transArray[trans_count].item_size++] = item_code_map[item];
            transaction_map[trans_no] = trans_count;
            trans_count++;
            //printf("Transaction Count :%d -> %d\n", trans_count, trans_no);
        }
        //else 
        {
            int _idx = transaction_map[trans_no]; 
            auto checkItemExist = [](Transaction* _tr, int _code) -> bool
            {
                bool ret = false;
                for (int idx = 0; idx < _tr->item_size; idx++) {
                    if (_tr->item_code[idx] == _code) return true;
                }
                return ret;
            };
            //if (!checkItemExist(&(transArray[_idx]), item_code_map[item])) 
            transArray[_idx].item_code[transArray[_idx].item_size++] = item_code_map[item];
        }

        if (trans_count >= TRANS_NUM) break;
        if (item_count >= item_num) break;
    }
    fs.close();
    
    printf("Item Count: %d, Transaction Count: %d\n", item_count, trans_count);
   
    size_t total = 0;
    auto begin = chrono::high_resolution_clock::now();

    /* sort item code array for each transaction */
    for (int _tr_idx = 0; _tr_idx < trans_count; _tr_idx++) {
        qsort(transArray[_tr_idx].item_code, transArray[_tr_idx].item_size, sizeof(int), itemcodeComp);
        int glb_i = 0;
        int i;
        for (i = 0; i < transArray[_tr_idx].item_size-1; i++) {
            while (i < transArray[_tr_idx].item_size && transArray[_tr_idx].item_code[i] == transArray[_tr_idx].item_code[i+1]) {
                i++;
            }
            transArray[_tr_idx].item_code[glb_i++] = transArray[_tr_idx].item_code[i];
        }
        if (i == transArray[_tr_idx].item_size-1) {
            transArray[_tr_idx].item_code[glb_i++] = transArray[_tr_idx].item_code[i-1];
        }
        transArray[_tr_idx].item_size = glb_i;
    }
    
    auto end = chrono::high_resolution_clock::now();
    total += chrono::duration_cast<chrono::milliseconds>(end-begin).count();
 
    auto printTrans = [](int _arr_size, Transaction* _trans_array)
    {
        for (int _tr_idx = 0; _tr_idx < _arr_size; _tr_idx++) {
            printf("Transaction %d:\n", _trans_array[_tr_idx].trans_no);
            for (int _it_idx = 0; _it_idx < _trans_array[_tr_idx].item_size; _it_idx++) {
                printf("\t Item %d\n", _trans_array[_tr_idx].item_code[_it_idx]); 
            }   
        }
    };
    //printTrans(trans_count, transArray);

    begin = chrono::high_resolution_clock::now();

    Item *itemArray = (Item*)malloc(item_count*sizeof(Item));
    memset(itemArray, 0, item_count*sizeof(Item));
    memory_use += item_count*sizeof(Item);
    for (int i = 0; i < item_count; i++) {
        itemArray[i].item_no = i;
    }
    
    /* request cuda memory */
    Transaction *dev_transArray = NULL;
    cudaMalloc(&dev_transArray, TRANS_NUM*sizeof(Transaction));
    cudaMemcpy(dev_transArray, transArray, TRANS_NUM*sizeof(Transaction), cudaMemcpyHostToDevice);
    dev_memory_use += TRANS_NUM*sizeof(Item);
    
    Item *dev_itemArray = NULL;
    cudaMalloc(&dev_itemArray, item_count*sizeof(Item));
    cudaMemcpy(dev_itemArray, itemArray, item_count*sizeof(Item), cudaMemcpyHostToDevice);
    dev_memory_use += item_count*sizeof(Item);
    
    /* calculate single item frequency */
    int num_threads_per_block = (num_threads < block_size) ? num_threads : block_size;
    //dim3 gridSize(num_threads/block_size);
    dim3 gridSize(NUM_THREADS/BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE);
    item_freq_count<<<gridSize, blockSize>>>(trans_count, dev_transArray, dev_itemArray);

    /* copy the results back to host */
    cudaMemcpy(itemArray, dev_itemArray, item_count*sizeof(Item), cudaMemcpyDeviceToHost);
    
    /* sort transaction array for each item */
    for (int _it_idx = 0; _it_idx < item_count; _it_idx++) {
        qsort(itemArray[_it_idx].trans_array, itemArray[_it_idx].trans_array_size, sizeof(int), itemcodeComp);
        int glb_i = 0;
        int i;
        for (i = 0; i < itemArray[_it_idx].trans_array_size-1; i++) {
            while (i < itemArray[_it_idx].trans_array_size && itemArray[_it_idx].trans_array[i] == itemArray[_it_idx].trans_array[i+1]) {
                i++;
            }
            itemArray[_it_idx].trans_array[glb_i++] = itemArray[_it_idx].trans_array[i];
        }
        if (i == itemArray[_it_idx].trans_array_size-1) {
            itemArray[_it_idx].trans_array[glb_i++] = itemArray[_it_idx].trans_array[i-1];
        }
        itemArray[_it_idx].trans_array_size = glb_i;
    }
    
    end = chrono::high_resolution_clock::now();
    total += chrono::duration_cast<chrono::milliseconds>(end-begin).count();

    /* check point of transposed database */
    auto printItems = [](int _arr_size, Item* _item_array)
    {
        for (int _it_idx = 0; _it_idx < _arr_size; _it_idx++) {
            printf("Item %d (freq %d):\n", _item_array[_it_idx].item_no, _item_array[_it_idx].freq);
            for (int _tr_idx = 0; _tr_idx < _item_array[_it_idx].trans_array_size; _tr_idx++) {
                printf("\t Transaction %d\n", _item_array[_it_idx].trans_array[_tr_idx]); 
            }   
        }
    };
    //printItems(item_count, itemArray);
    
    begin = chrono::high_resolution_clock::now();
 
    /* start to prune */
    int globalIdx = 0;
    int *dev_globalIdx = NULL;
    cudaMalloc(&dev_globalIdx, sizeof(int));
    cudaMemcpy(dev_globalIdx, &globalIdx, sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_itemArray, itemArray, item_count*sizeof(Item), cudaMemcpyHostToDevice);

    ItemSet *itemsetArray = (ItemSet*)malloc(item_count*sizeof(ItemSet));
    memset(itemsetArray, 0, item_count*sizeof(ItemSet));
    memory_use += item_count*sizeof(ItemSet);
    
    ItemSet *dev_itemsetArray = NULL;
    cudaMalloc(&dev_itemsetArray, item_count*sizeof(ItemSet));
    cudaMemcpy(dev_itemsetArray, itemsetArray, item_count*sizeof(ItemSet), cudaMemcpyHostToDevice);
    dev_memory_use += item_count*sizeof(ItemSet);
    
    /* kernel doing selection for single item with minimum support */
    select_with_min_support<<<gridSize, blockSize>>>(item_count, dev_itemArray, min_support, dev_itemsetArray, dev_globalIdx);

    cudaMemcpy(itemsetArray, dev_itemsetArray, item_count*sizeof(ItemSet), cudaMemcpyDeviceToHost);
    cudaMemcpy(&globalIdx, dev_globalIdx, sizeof(int), cudaMemcpyDeviceToHost);
   
    free(itemArray);
    
    end = chrono::high_resolution_clock::now();
    total += chrono::duration_cast<chrono::milliseconds>(end-begin).count();

    /* check point of transposed database */
    auto printItemSet = [](int _arr_size, ItemSet* _itemset_array)
    {
        for (int _it_idx = 0; _it_idx < _arr_size; _it_idx++) {
            printf("ItemSet %d (size %d):\n", _it_idx, _itemset_array[_it_idx].item_set_size);
            for (int i = 0; i < _itemset_array[_it_idx].item_set_size; i++) {
                printf("\tItem %d", _itemset_array[_it_idx].item_set_code[i]);
            }
            printf("\n");
            for (int i = 0; i < _itemset_array[_it_idx].trans_array_size; i++) {
                printf("\tTransaction %d", _itemset_array[_it_idx].trans_array[i]);
            }  
            printf("\n");
            printf("\tSet Index (%d,%d)\n", _itemset_array[_it_idx].set1_index, _itemset_array[_it_idx].set2_index);
        }
    };
    //printItemSet(globalIdx, itemsetArray);

 
    /* Record in Support Count */
    auto sc_record_func = [](vector<SupportCount>& vec, int itemset_count, ItemSet* itemset_array)
    {
        for (int is_idx = 0; is_idx < itemset_count; is_idx++) {
            SupportCount sc;
            sc.freq = itemset_array[is_idx].freq;
            sc.count = itemset_array[is_idx].item_set_size;
            sc.item_code_array = (int*)malloc(sc.count * sizeof(int));
            memcpy(sc.item_code_array, itemset_array[is_idx].item_set_code, sc.count*sizeof(int));
            qsort(sc.item_code_array, sc.count, sizeof(int), itemcodeComp);
            vec.push_back(sc);
        }
    };   
    
    sc_record_func(support_count_vec, item_count, itemsetArray);

    /* Now we get the transposed database that every item set with size 1 has a corresponding list of transactions */
    /* Generate itemset with size 2 */
    
    int cardinality = 2;
    int currSetSize = globalIdx;
    int candidateSetSize = 0;
    int *dev_candidateSetSize = NULL;
    ItemSet* currSet = itemsetArray;
    ItemSet* dev_currSet = NULL;
    ItemSet* candidateSet = NULL;
    ItemSet* dev_candidateSet = NULL;

    cudaMalloc(&dev_candidateSetSize, sizeof(int));
    
    //cudaMalloc(&dev_currSet, currSetSize*sizeof(ItemSet));
    //cudaMemcpy(dev_currSet, currSet, currSetSize*sizeof(ItemSet), cudaMemcpyHostToDevice);

    while (true) {
        candidateSet = (ItemSet*)genNextItemSetArray(currSetSize, currSet, cardinality, &candidateSetSize);
        if (candidateSetSize == 0 || candidateSet == NULL) {
            break;
        }
        assert(candidateSet != NULL);          
        
        printf("\n\n Next candidate size is %d\n", candidateSetSize);
        memory_use += candidateSetSize*sizeof(ItemSet);
        
        begin = chrono::high_resolution_clock::now();

        /* allocate GPU kernel memory */
        cudaMemcpy(dev_candidateSetSize, &candidateSetSize, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_globalIdx, &globalIdx, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_currSet, currSetSize*sizeof(ItemSet));
        assert(dev_currSet != NULL);
        cudaMemcpy(dev_currSet, currSet, currSetSize*sizeof(ItemSet), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_candidateSet, candidateSetSize*sizeof(ItemSet));
        cudaMemcpy(dev_candidateSet, candidateSet, candidateSetSize*sizeof(ItemSet), cudaMemcpyHostToDevice);        
        
        dev_memory_use += currSetSize*sizeof(ItemSet);
        dev_memory_use += candidateSetSize*sizeof(ItemSet);

        /* launch the kernel */
        dim3 gSize(NUM_THREADS/BLOCK_SIZE);
        dim3 bSize(BLOCK_SIZE);
        find_support_count<<<gSize, bSize>>>(candidateSetSize, 
                                                     dev_candidateSet, 
                                                     dev_globalIdx, 
                                                     dev_currSet, 
                                                     dev_transArray,
                                                     min_support);

        /* copy the result back */
        cudaMemcpy(candidateSet, dev_candidateSet, candidateSetSize*sizeof(ItemSet), cudaMemcpyDeviceToHost);
        
        end = chrono::high_resolution_clock::now();
        total += chrono::duration_cast<chrono::milliseconds>(end-begin).count(); 
        
        /* prune if freq == -1 */
        int _glb_set_idx = 0;
        for (int set_idx = 0; set_idx < candidateSetSize; set_idx++) {
            if (!candidateSet[set_idx].pruned) {
                //printf("---prune candidate %d freq %d\n", set_idx, candidateSet[set_idx].freq);
                memcpy(&(candidateSet[_glb_set_idx++]), &(candidateSet[set_idx]), sizeof(ItemSet));
            }
        }
        //candidateSetSize = _glb_set_idx;
        //printItemSet(_glb_set_idx, candidateSet); 

        /* Make statistics for support count */
        sc_record_func(support_count_vec, candidateSetSize, candidateSet);

        /* update the parameters and free previously used memory */
        free(currSet);
        cudaFree(dev_currSet);
        
        cardinality++;
        currSet = candidateSet;
        currSetSize = candidateSetSize;
        //dev_currSet = dev_candidateSet; 
        cudaFree(dev_candidateSet);
        globalIdx = 0;

        if (_glb_set_idx <= 1) {
            break;
        }
        
        max_memory_use = (max_memory_use < memory_use) ? memory_use : max_memory_use;
        max_dev_memory_use = (max_dev_memory_use < dev_memory_use) ? dev_memory_use : max_dev_memory_use;
        memory_use -= currSetSize*sizeof(ItemSet);
        dev_memory_use -= candidateSetSize*sizeof(ItemSet);
    }
    
    /* final result */
    //printItemSet(currSetSize, currSet); 
    
    /* Finally generate association rules */
    auto get_support_count = [](vector<SupportCount>& vec, ItemSet* itemset)->int
    {
        int _size = itemset->item_set_size;
        for (auto sc : vec) {
            if (sc.count != _size) continue;
            if (memcmp(itemset->item_set_code, sc.item_code_array, _size*sizeof(int)) != 0) continue;
            return sc.freq;
        }
        return 0;
    };
    
    function<void(ItemSet*, int, int, int, ItemSet*, vector<SupportCount>&)> get_rules_per_size;
    get_rules_per_size = [&get_support_count, &get_rules_per_size](ItemSet* sub_itemset, int array_index, int size, int start_pos, ItemSet* itemset, vector<SupportCount>& vec)
    {
        sub_itemset->item_set_code[array_index] = itemset->item_set_code[start_pos];
        if (array_index+1 == size) {
            int _support_count = get_support_count(vec, sub_itemset);
            /* now we can calculate the confidence */
            if (_support_count == 0) return;
            //printf("freq %f, s_count %f\n", (float)(itemset->freq), (float)(_support_count));
            float confidence =  (float)(itemset->freq) / (float)(_support_count);
            
            //printf("-----------------Association Rules--------------------\n");
            
            //printf("Items: \n");
            //for (int i = 0; i < size; i++) printf("\tItem %d\t", sub_itemset->item_set_code[i]);
            //printf("\nBase: \n");
            //for (int i = 0; i < itemset->item_set_size; i++) printf("\tItem %d\t", itemset->item_set_code[i]);
            //printf("\n\n=====  Confidence %f =====\n", confidence);

            //printf("------------------------------------------------------\n");
            
            return;
        }
        for (int next_pos = start_pos+1; next_pos < itemset->item_set_size; next_pos++) {
            get_rules_per_size(sub_itemset, array_index+1, size, next_pos, itemset, vec);
        }
    };

    
    auto getRules = [&get_rules_per_size](ItemSet* itemset, int size, vector<SupportCount>& vec)
    {
        //int *_code_array = (int*)malloc(size*sizeof(int));
        ItemSet *sub_itemset = (ItemSet*)malloc(sizeof(ItemSet));
        memset(sub_itemset, 0, sizeof(ItemSet));
        sub_itemset->item_set_size = size;
        
        int array_index = 0;
        for (int start_pos = 0; start_pos < itemset->item_set_size; start_pos++) {
            get_rules_per_size(sub_itemset, array_index, size, start_pos, itemset, vec);
        }

        free(sub_itemset);
    };

    for (int idx = 0; idx < currSetSize; idx++) {
        ItemSet* item_set = &currSet[idx];
        for (int _size = 1; _size <= item_set->item_set_size; _size++) {
            //getRules(item_set, _size, support_count_vec);
        }
    }
    
    cudaProfilerStop();    
 
    printf("Sumary : Item Count %d --- Trans Count %d\nExec Time %llu ms\n", item_count, trans_count, total);
    printf("\t CPU memory max usage : %llu bytes\n", max_memory_use);
    printf("\t GPU memory max usage : %llu bytes\n", max_dev_memory_use);

    return 0;
}
