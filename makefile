CC=g++
CFLAGS= -std=c++11 -O2 -g -fpermissive -c
OBJECTS= la_pack.o analysis.o filter.o func.o testSuite.o quaternion.o Source.o
PROG= prog


$(PROG) : $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

%.o: %.cpp matrix.h
	$(CC) $(CFLAGS) $<

clean:
	rm $(PROG) $(OBJECTS)
