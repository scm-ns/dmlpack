CC=g++
CFLAGS= -std=c++11 -c
DEBUG_FLAGS= -ggdb
OPTIMIZE_FLAG= 
LIB_FLAGS= -I/usr/include/python2.7
LIB= -lpython2.7
OBJECTS= data_source.o test.o
PROG= dml.exe
PROG_DEBUG= dml_debug.exe

$(PROG) : $(OBJECTS)
	$(CC) $(OBJECTS) $(LIB) $(LIB_FLAGS) -o $@

%.o: %.cpp matrix.h dmlpack.h
	$(CC) $(OPTIMIZE_FLAG) $(DEBUG_FLAGS) $(LIB_FLAGS) $(CFLAGS) $<

debug: 
	$(CC) $(OBJECTS) $(LIB) $(LIB_FLAGS) -o $@

clean:
	rm $(PROG_DEBUG) $(OBJECTS) *.gch
run:
	./$(PROG_DEBUG)
