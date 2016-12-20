CC=g++
CFLAGS= -std=c++11 -c
DEBUG_FLAGS= 
OPTIMIZE_FLAG= 
LIB_FLAGS= -I/usr/include/python2.7
LIB= -lpython2.7
OBJECTS= data_source.o test.o
PROG_DEBUG= dml_debug.exe



debug: $(PROG_DEBUG)

$(PROG_DEBUG) : $(OBJECTS)
	$(CC) $(OBJECTS) $(LIB) $(LIB_FLAGS) -o $@
	./$(PROG_DEBUG)

%.o: %.cpp matrix.h dmlpack.h
	$(CC) $(DEBUG_FLAGS) $(OPTIMIZE_FLAG) $(LIB_FLAGS) $(CFLAGS) $<

clean:
	rm $(PROG_DEBUG) $(OBJECTS) *.gch
