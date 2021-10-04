typedef struct _32bit_struct
{
	int bit1;
	int bit2;
	int bit3;
	int bit4;
	int bit5;
	int bit6;
	int bit7;
	int bit8;
	int bit9;
	int bit10;
	int bit11;
	int bit12;
	int bit13;
	int bit14;
	int bit15;
	int bit16;
	int bit17;
	int bit18;
	int bit19;
	int bit20;
	int bit21;
	int bit22;
	int bit23;
	int bit24;
	int bit25;
	int bit26;
	int bit27;
	int bit28;
	int bit29;
	int bit30;
	int bit31;
	int bit32;
}_32it_struct;


struct struct_udp
{
    uint32_t Label367; 				// START
    uint32_t Label366_DTIF_Header;	//

//////  OWNSHIP INFORMATION  //////
    uint32_t DTIF_Packet_Header;	// OWNSHIP START
    
	uint32_t TYPE_0000_0;			//OWNSHIP INFORMATION_3char
    uint32_t TYPE_0000_1;			//OWNSHIP INFORMATION_3char
    uint32_t TYPE_0000_2;			//OWNSHIP INFORMATION_2char

    uint32_t TYPE_0001_0;
    uint32_t TYPE_0001_1;

    uint32_t TYPE_0002_0;
    uint32_t TYPE_0002_1;
    uint32_t TYPE_0002_2;

	uint32_t TYPE_0003_0;
    uint32_t TYPE_0003_1;

	// uint32_t TYPE_0004_0;
    // uint32_t TYPE_0004_1;

	// uint32_t TYPE_0005_0;

	// uint32_t TYPE_0007_0;
    // uint32_t TYPE_0007_1;

///// END OF OWNSHIP INFORMATION /////

/////  INTRUDER INFORMATION //////

    uint32_t INTR_DTIF_Packet_Header;
    
	uint32_t INTR_TYPE_0000_0;			
    uint32_t INTR_TYPE_0000_1;			
    uint32_t INTR_TYPE_0000_2;			

    uint32_t INTR_TYPE_0001_0;
    uint32_t INTR_TYPE_0001_1;

    // uint32_t INTR_TYPE_0002_0;
    // uint32_t INTR_TYPE_0002_1;
    // uint32_t INTR_TYPE_0002_2;

	uint32_t INTR_TYPE_0003_0;
    uint32_t INTR_TYPE_0003_1;

	// uint32_t INTR_TYPE_0004_0;
    // uint32_t INTR_TYPE_0004_1;

	// uint32_t INTR_TYPE_0005_0;

	// uint32_t INTR_TYPE_0007_0;
    // uint32_t INTR_TYPE_0007_1;

///// END OF INTRUDER INFORMATION /////

    uint32_t Label367_END; // END
};





// struct struct_udp_test
// {
//     uint32_t a0;
//     uint32_t b0;
//     uint32_t c0;
// 	uint32_t d0;
	
// 	uint32_t a1;
//     uint32_t b1;
//     uint32_t c1;
// 	uint32_t d1;

//     uint32_t a2;
//     uint32_t b2;
//     uint32_t c2;
// 	uint32_t d2;

//     uint32_t a3;
//     uint32_t b3;
//     uint32_t c3;
// 	uint32_t d3;

//     uint32_t a4;
//     uint32_t b4;


// };