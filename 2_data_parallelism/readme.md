### notes

2 types of parallelism\
a. task parallelism modest parallelism - diff oper on same or dff data\
b. data parallelism highly parallelism - same oper on large different data - most suitable for gpus

Vector addition - Hello world of data parallel computing

System organization\
a. CPU (host) -> Main memory (host mem) \
b. GPU (device) -> GPU mem (device/global mem)\
CPU and GPU have seperate memory therefor cannot access each others memories\ 
need to transfer data between them