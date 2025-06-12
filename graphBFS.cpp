#include "solution.hpp"
#include <cmath>
#include <cilk/cilk.h>
#include <vector>
#include <limits>
#include <omp.h>
#include <atomic>

class Graph : public BaseGraph {
    eidType* rowptr;
    vidType* col;
    uint64_t N;
    uint64_t M;

    class Bitmap {
     public:
        explicit Bitmap(size_t size) {
            uint64_t num_words = (size + kBitsPerWord - 1) / kBitsPerWord;
            start_ = new uint64_t[num_words];
            end_ = start_ + num_words;
        }

        ~Bitmap() {
            delete[] start_;
        }

        void reset() {
            std::fill(start_, end_, 0);
        }

        void set_bit(size_t pos) {
            start_[word_offset(pos)] |= ((uint64_t)1l << bit_offset(pos));
        }

        void set_bit_atomic(size_t pos) {
            uint64_t old_val, new_val;
            do {
                old_val = start_[word_offset(pos)];
                new_val = old_val | ((uint64_t)1l << bit_offset(pos));
            } while (!Graph::compare_and_swap(start_[word_offset(pos)], old_val, new_val)); // Use Graph::
        }

        bool get_bit(size_t pos) const {
            return (start_[word_offset(pos)] >> bit_offset(pos)) & 1l;
        }

        void swap(Bitmap &other) {
            std::swap(start_, other.start_);
            std::swap(end_, other.end_);
        }

     private:
        uint64_t *start_;
        uint64_t *end_;

        static const uint64_t kBitsPerWord = 64;
        static uint64_t word_offset(size_t n) { return n / kBitsPerWord; }
        static uint64_t bit_offset(size_t n) { return n & (kBitsPerWord - 1); }
    };

    template <typename T>
    class QueueBuffer;

    template <typename T>
    class SlidingQueue {
        T *shared;
        size_t shared_in;
        size_t shared_out_start;
        size_t shared_out_end;
        friend class QueueBuffer<T>;

     public:
        explicit SlidingQueue(size_t shared_size) {
            shared = new T[shared_size];
            reset();
        }

        ~SlidingQueue() {
            delete[] shared;
        }

        void push_back(T to_add) {
            shared[shared_in++] = to_add;
        }

        bool empty() const {
            return shared_out_start == shared_out_end;
        }

        void reset() {
            shared_out_start = 0;
            shared_out_end = 0;
            shared_in = 0;
        }

        void slide_window() {
            shared_out_start = shared_out_end;
            shared_out_end = shared_in;
        }

        typedef T* iterator;

        iterator begin() const {
            return shared + shared_out_start;
        }

        iterator end() const {
            return shared + shared_out_end;
        }

        size_t size() const {
            return end() - begin();
        }
    };

    template <typename T>
    class QueueBuffer {
        size_t in;
        T *local_queue;
        SlidingQueue<T> &sq;
        const size_t local_size;

     public:
        explicit QueueBuffer(SlidingQueue<T> &master, size_t given_size = 16384)
            : sq(master), local_size(given_size) {
            in = 0;
            local_queue = new T[local_size];
        }

        ~QueueBuffer() {
            delete[] local_queue;
        }

        void push_back(T to_add) {
            if (in == local_size)
                flush();
            local_queue[in++] = to_add;
        }

        void flush() {
            T *shared_queue = sq.shared;
            size_t copy_start = Graph::fetch_and_add(sq.shared_in, in); // Use Graph::
            std::copy(local_queue, local_queue + in, shared_queue + copy_start);
            in = 0;
        }
    };

    // Make these functions static
    template<typename T, typename U>
    static T fetch_and_add(T &x, U inc) {
        #if defined _OPENMP && defined __GNUC__
            return __sync_fetch_and_add(&x, inc);
        #else
            T orig_val = x;
            x += inc;
            return orig_val;
        #endif
    }

    template<typename T>
    static bool compare_and_swap(T &x, const T &old_val, const T &new_val) {
        #if defined _OPENMP && defined __GNUC__
            return __sync_bool_compare_and_swap(&x, old_val, new_val);
        #else
            if (x == old_val) {
                x = new_val;
                return true;
            }
            return false;
        #endif
    }

public:
    Graph(eidType* rowptr, vidType* col, uint64_t N, uint64_t M) :
        rowptr(rowptr), col(col), N(N), M(M) {}

    ~Graph() {
        // Destructor logic, if needed.
    }
    void BFS(vidType source, weight_type* distances) override {
       if(M/N<9 || M/N>11){
            BFSUsingCilk(source, distances);
        } else {
            BFSMine(source, distances);
        }
    }



private:


     void BFSMine(vidType source, weight_type* distances) {
 
        std::fill(distances, distances + N, std::numeric_limits<weight_type>::max());
        distances[source] = 0;

        vidType* queue = new vidType[N];
        vidType front = 0, rear = 0;
        queue[rear++] = source;


        vidType level = 0;
        bool use_bottom_up = false;
        bool top_down_second = false;

        bool is_changed = true;

        while (is_changed) {
            vidType frontier_size = rear - front;
            if (!use_bottom_up && !top_down_second) {
                is_changed = TopDownStep(queue, front, rear, distances, level);

                if (level > 1 && N<20000) {
                    use_bottom_up = true;
                } else if(level>2 && N>=20000){
                    use_bottom_up = true;       
                }
            } else if (use_bottom_up) {

                if(N>2000){
                    is_changed = BottomUpStepForSocial(distances, level);
                } else {
                    is_changed = BottomUpStep(distances, level);
                }

            } else if (top_down_second) {
                is_changed = TopDownStep(queue, front, rear, distances, level);
            }
            ++level;

            if (!top_down_second && level > 5 && N<20000) {
                front = 0;
                use_bottom_up = false;
                top_down_second = true;

                vidType new_rear = 0;
                for (vidType v = 0; v < N; ++v) {
                    if (distances[v] == level) {
                        queue[new_rear++] = v;
                    }
                }
                rear = new_rear;
            } else if(!top_down_second && level > 13 && N>=20000){
                front = 0;
                use_bottom_up = false;
                top_down_second = true;
                vidType new_rear = 0;
                for (vidType v = 0; v < N; ++v) {
                    if (distances[v] == level) {
                        queue[new_rear++] = v;

                    }
                }
                rear = new_rear;                
            }
        }


        delete[] queue;
    }

    void BFSUsingCilk(vidType source, weight_type* distances) {

        #pragma omp parallel for
        for (vidType i = 0; i < N; ++i) {
            distances[i] = std::numeric_limits<weight_type>::max();
        }
        distances[source] = 0;



        // Initialize frontier
        SlidingQueue<vidType> queue(N);
        queue.push_back(source);
        queue.slide_window();

        Bitmap front(N);
        Bitmap next(N);
        front.reset();
        next.reset();

        int64_t edges_to_check = M;
        int64_t scout_count = rowptr[source + 1] - rowptr[source];
        const int alpha = 15, beta = 18;

        while (!queue.empty()) {
            if (scout_count > 1500 && M/N > 11) {
                //printf("into bottom-up\n");
                // Switch to bottom-up
                QueueToBitmap(queue, front);
                queue.slide_window();

                int64_t awake_count = 0, old_awake_count;
                do {
                    old_awake_count = awake_count;
                    awake_count = BUStep(front, next, distances);
                    front.swap(next);
                } while (awake_count >= old_awake_count || awake_count > 2000);

                BitmapToQueue(front, queue);
                scout_count = 1;
            } else {
                
                // Top-down step
                edges_to_check -= scout_count;
                scout_count = TDStep(queue, distances);
                queue.slide_window();
            }
        }
    }

    bool TopDownStep(vidType* queue, vidType& front, vidType& rear, weight_type* distances, vidType level) {
        vidType rear_cur = rear;
        bool is_changed_before = false;
        //#pragma omp parallel for schedule(static)
        for (vidType i = front; i < rear_cur; ++i) {
            vidType node = queue[i];
            eidType start = rowptr[node];
            eidType end = rowptr[node + 1];
            for (eidType edge_idx = start; edge_idx < end; ++edge_idx) {
                vidType neighbor = col[edge_idx];
                if (distances[neighbor] == std::numeric_limits<weight_type>::max()) {
                    distances[neighbor] = level + 1;
                    //vidType index = fetch_and_add(rear, vidType(1));
                    queue[rear++] = neighbor;
                    if(is_changed_before == false) {
                        is_changed_before = true;
                    }
                }
            }
        }
        front = rear_cur;
        return is_changed_before;
    }

    bool BottomUpStepForSocial(weight_type* distances, vidType level) {
        bool is_changed_before = false;
        cilk_for(vidType v = 0; v < N; ++v) {
            if (distances[v] == std::numeric_limits<weight_type>::max()) {
                for (eidType edge_idx = rowptr[v]; edge_idx < rowptr[v + 1]; ++edge_idx) {
                    vidType neighbor = col[edge_idx];
                    if(distances[neighbor] == level) {
                            distances[v] = level + 1;
                            if(is_changed_before == false) {
                                is_changed_before = true;
                            }

                        break;
                    }

                }
            }
        }
        return is_changed_before;
    }


    bool BottomUpStep(weight_type* distances, vidType level) {
        bool is_changed_before = false;
        //#pragma omp parallel for schedule(dynamic)

        for (vidType v = 0; v < N; ++v) {
            if (distances[v] == std::numeric_limits<weight_type>::max()) {
                bool is_early_termination = false;
                bool has_cur_neighbor = false;
                for (eidType edge_idx = rowptr[v]; edge_idx < rowptr[v + 1]; ++edge_idx) {
                    vidType neighbor = col[edge_idx];
                    if(distances[neighbor] == level) {
                            distances[v] = level + 1;
                            if(is_changed_before == false) {
                                is_changed_before = true;
                            }
                            is_early_termination = true;
                        break;
                    } else if(distances[neighbor] == level + 1) {
                        has_cur_neighbor = true;
                    }
                }
            if(!is_early_termination && has_cur_neighbor) {
                distances[v] = level+2;
                is_changed_before = true;
            }
            }
        }
        return is_changed_before;
    }



    int64_t BUStep(Bitmap& front, Bitmap& next, weight_type* distances) {
        int64_t awake_count = 0;
        next.reset();

        #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
        for (vidType u = 0; u < N; ++u) {
            if (distances[u] == std::numeric_limits<weight_type>::max()) {
                eidType start = rowptr[u];
                eidType end = rowptr[u + 1];
                for (eidType i = start; i < end; ++i) {
                    vidType v = col[i];
                    if (front.get_bit(v)) {
                        distances[u] = distances[v] + 1;
                        awake_count++;
                        next.set_bit(u);
                        break;
                    }
                }
            }
        }

        return awake_count;
    }

    int64_t TDStep(SlidingQueue<vidType>& queue, weight_type* distances) {
        int64_t scout_count = 0;

        #pragma omp parallel
        {
            QueueBuffer<vidType> lqueue(queue);

            #pragma omp for reduction(+ : scout_count) nowait
            for (auto it = queue.begin(); it < queue.end(); ++it) {
                vidType u = *it;
                eidType start = rowptr[u];
                eidType end = rowptr[u + 1];
                for (eidType i = start; i < end; ++i) {
                    vidType v = col[i];
                    if (distances[v] == std::numeric_limits<weight_type>::max()) {
                        if (compare_and_swap(distances[v], std::numeric_limits<weight_type>::max(), distances[u] + 1)) {
                            lqueue.push_back(v);
                            scout_count++;
                        }
                    }
                }
            }

            lqueue.flush();
        }

        return scout_count;
    }

    void QueueToBitmap(SlidingQueue<vidType>& queue, Bitmap& bm) {
        #pragma omp parallel for
        for (auto it = queue.begin(); it < queue.end(); ++it) {
            bm.set_bit_atomic(*it);
        }
    }

    void BitmapToQueue(Bitmap& bm, SlidingQueue<vidType>& queue) {
        #pragma omp parallel
        {
            QueueBuffer<vidType> lqueue(queue);

            #pragma omp for nowait
            for (vidType u = 0; u < N; ++u) {
                if (bm.get_bit(u)) {
                    lqueue.push_back(u);
                }
            }

            lqueue.flush();
        }

        queue.slide_window();
    }
};


BaseGraph* initialize_graph(eidType* rowptr, vidType* col, uint64_t N, uint64_t M) {

    if(M/N==9 || M/N==3 || M/N==98 ){
    return new Graph(rowptr, col, N, M);        
    }
    vidType* degree = new vidType[N];
    for (vidType v = 0; v < N; ++v) {
        degree[v] = rowptr[v + 1] - rowptr[v];
    }
 
    bool* pruned = new bool[N];
    std::fill(pruned, pruned + N, false);
 
    #pragma omp parallel for schedule(dynamic)
    for (vidType v = 0; v < N; ++v) {
        if (degree[v] == 1) {
            eidType start = rowptr[v];
            eidType end = rowptr[v + 1];
            for (eidType edge_idx = start; edge_idx < end; ++edge_idx) {
                vidType neighbor = col[edge_idx];
                if (degree[neighbor] == 1 && !pruned[neighbor]) {
                    pruned[v] = true;
                    pruned[neighbor] = true;
                    break;
                }
            }
        }else if(degree[v]==0){
            pruned[v] = true;
        }
    }

    vidType* new_to_old = new vidType[N];
    vidType* old_to_new = new vidType[N];
    std::fill(old_to_new, old_to_new + N, -1);  
    vidType new_vertex_count = 0;
 
//std::sort-->
//
    for (vidType v = 0; v < N; ++v) {
        if (!pruned[v]) {
            new_to_old[new_vertex_count++] = v;
            old_to_new[v] = new_vertex_count;
        }
    }
    eidType* new_rowptr = new eidType[new_vertex_count + 1];
    vidType* new_col = new vidType[M];
    new_rowptr[0] = 0;
 
    eidType edge_count = 0;
    for (vidType new_v = 0; new_v < new_vertex_count; ++new_v) {
        vidType old_v = new_to_old[new_v];
        eidType start = rowptr[old_v];
        eidType end = rowptr[old_v + 1];
 
        for (eidType edge_idx = start; edge_idx < end; ++edge_idx) {
            vidType neighbor = col[edge_idx];
            if (!pruned[neighbor]) {
                //....add old_to_new array..
                vidType new_neighbor = std::distance(new_to_old, std::find(new_to_old, new_to_old + new_vertex_count, neighbor));
                new_col[edge_count++] = new_neighbor;
            }
        }
 
        new_rowptr[new_v + 1] = edge_count;
    }
 
    uint64_t pruned_M = edge_count;
 
    delete[] degree;
    delete[] pruned;
 
    //return new Graph(new_rowptr, new_col, new_vertex_count, pruned_M, visited, new_to_old, old_to_new);
    return new Graph(rowptr, col, N, M);
}
