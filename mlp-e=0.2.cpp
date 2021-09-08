// "The automatic parameter-exploration in principle of machine-learning: Powering the evolutionary modeling on the origin of life"
// By Yuzhen Liang, Chunwu Yu and Wentao Ma*.
// C source code for the program
// The version correponds to the red-line case in Fig.2a of the article
// This code is based upon the model established in our previous study:"Nucleotide synthetase ribozymes may have emerged first in the RNA World" (2007)
// *** Please pay attention to the annotations starting with stars, which are key codes associated with the machine-learning method used in the present study

#include "stdafx.h"

using namespace System;

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <conio.h>

// RANDOM NUMBER GENERATOR BY ZIFF //
#define A1 471
#define B1 1586
#define C1 6988
#define D1 9689
#define M 16383
#define RIMAX 2147483648.0        // = 2^31 
#define RandomInteger (++nd, ra[nd & M] = ra[(nd-A1) & M] ^ ra[(nd-B1) & M] ^ ra[(nd-C1) & M] ^ ra[(nd-D1) & M])
void seed(long seed);  // random number initialization 
static long ra[M + 1], nd;
/////////////////////////////////////

#define LEN sizeof(struct rna)
#define C 2
#define G 3
#define A 1
#define U 4
#define STEPNUM 150000    // Total time steps of Monte Carlo simulation
#define STAREC 0          // The step to start record
#define RECINT 5000       // The interval steps of recording
#define MAX_RNA_LENGTH 100    // Defining maximum RNA length allowed in the simulation

#define SD 555
#define SIDE 20                // The side length of the two-dimensional grid
#define TOTAL_MATERIAL 40000   // Total materials in the system
#define NSRSEQ A,G,C,A,U,G,C,U     // The assumed specific sequence with which a polynucleotide could act as an nt-synthetase ribozyme (NSR)
#define CONTRSEQ1 C,U,A,C,G,U,A,G  // The assumed control sequence without function
#define CONTRSEQ2 C,G,U,U,A,A,C,G  // 
#define CONTRSEQ3 A,U,C,G,C,G,A,U  // 

#define INOCUSEQ NSRSEQ//
#define INOCUNUM 100
#define INOCUSEQ1 CONTRSEQ1//
#define INOCUNUM1 0
#define INOCUSEQ2 CONTRSEQ2 //  
#define INOCUNUM2 0
#define INOCUSEQ3 CONTRSEQ3 //  
#define INOCUNUM3 0
#define INOCUSTEP 10000
#define PSBP 0.9           // Probability of separation of a base-pair
#define PLT 0.9            // Probability of a template-directed ligation 
#define PMN (PMV/2)        // Probability of movement of a nucleotide
#define TNSS 1             // Turn of nt-synthesis by NSR each step
#define FDMOV (pow(p->length1+p->length2,1/3.0))  // The factor defining the relationship betwlearning_raten probability of moving and molecular weight
#define ROOMNUM (SIDE*SIDE)   // Room numbers in the grid

#define LEARNING_ROUND 400 //*** The upper limit of learning rounds in the machine-learning
double learning_rate=0.2;  //*** The learning rate in the machine-learning

//******The parameters that are automatically explored in the machine-learning
double PNF;   // Probability of nucleotide formation (not catalyzed by ribozyme)
double PNFR;  // Probability of nucleotide formation under the catalysis of NSR
double PND;   // Probability of nucleotide decay
double PRL;   // Probability of the random ligation of nucleotides and oligonucleotides 
double PBB;   // Probability of breaking of a phosphodiester bond
double PAT;   // Probability of attracting a substrate by a template
double PFP;   // Probability of false base-pairing
double PMV;   // Probability of the movement of raw material to an adjacent grid room
//*************************************************************************************

long randl(long);      // random number betwlearning_raten 0 and parameter 
double randd(void);    // random double betwlearning_raten 0 and 1         
void avail_xy_init(void); // Initialization for xy_choose
void xy_choose(void);     // Picks a room at random 
void fresh_unit(void);    // Updating a unit for the next time step
int findseq(char seq[], int seqlength, struct rna *p, int m, int n); //find a specific subsequence in a sequence 
void inits(void);         // initialization of the system
void inoculate(void);
void unit_case(void);     // Action of units (molecules) in the system
int record(void);         // Data recording
void save_result(void);   // Data saving 
void freepool(void);      // Memory releasing
double max_fabs(double a, double b, double c, double d,	double e, double f,	double g, double h);

struct rna                // A unit of nucleotide or polynucleotide
{
	char information[2][MAX_RNA_LENGTH];
	int length1;
	int length2;
	int nick;
	struct rna *next;
	struct rna *prior;
};
struct rna *room_head[2][SIDE][SIDE];
struct rna *p, *p1, *p2, *p3, *p4, *ps, *ps1, *ps2;

static char nsrseq[50]    = { NSRSEQ };        
static char contrseq1[50] = { CONTRSEQ1 };
static char contrseq2[50] = { CONTRSEQ2 };
static char contrseq3[50] = { CONTRSEQ3 };
static char inocuseq[50]  = { INOCUSEQ };
static char inocuseq1[50] = { INOCUSEQ1 };
static char inocuseq2[50] = { INOCUSEQ2 };
static char inocuseq3[50] = { INOCUSEQ3 };
static int raw_arr[SIDE][SIDE];

int over_max_len = 0;
int x, y;                 // The coordinate of rooms in the grid 
int synlength, contrlength1, contrlength2, contrlength3;
int inoculength, inoculength1, inoculength2, inoculength3;
int randcase, randcase1, randcaser, randcaser1, length3, g, h = 0, g_end = 0, gi, g_stop = 0;
int flag, flag1, flag2, flag3, flagn, flagn1, flagn2, flagn3, flag4 = 0, flag5;
long i;                  // Cycle variable for Monte Carlo steps
long available;
long availabl[ROOMNUM];
long recstep[(STEPNUM - STAREC) / RECINT + 1];    // For recording steps
float NSR_num[(STEPNUM - STAREC) / RECINT + 1];   // For recording the number of NSR in steps
float contr_num1[(STEPNUM - STAREC) / RECINT + 1];  // For recording the number of Ctrl molecules in steps
float contr_num2[(STEPNUM - STAREC) / RECINT + 1];  // 
float contr_num3[(STEPNUM - STAREC) / RECINT + 1];  // 
float total_mat_num[(STEPNUM - STAREC) / RECINT + 1];  // For recording the number of total materials in steps
float unit_num[(STEPNUM - STAREC) / RECINT + 1];  // For recording the number of units in steps
float raw_num[(STEPNUM - STAREC) / RECINT + 1];  // For recording the number of raw in steps

double PARA[2][9][LEARNING_ROUND+1];           // *** For recording parameter values during the learning
double deltaPARA[9][LEARNING_ROUND+1];         // *** For recording the change of NSR with the change of a parameter during the learning
double NSR_num_PARA[9][LEARNING_ROUND+1];      // *** For recording the number of NSR during the learning

void seed(long seed)	  // Random number initialization
{
	int a;

	if (seed<0) { puts("seed error."); exit(1); }
	ra[0] = (long)fmod(16807.0*(double)seed, 2147483647.0);
	for (a = 1; a <= M; a++)
	{
		ra[a] = (long)fmod(16807.0 * (double)ra[a - 1], 2147483647.0);
	}
}

//------------------------------------------------------------------------------
long randl(long num)      // Random integer number betwlearning_raten 0 and num-1 
{
	return(RandomInteger % num);
}

//------------------------------------------------------------------------------
double randd(void)        // Random real number betwlearning_raten 0 and 1 
{
	return((double)RandomInteger / RIMAX);
}

//------------------------------------------------------------------------------
void avail_xy_init(void)   // Initialization for xy_choose
{
	int j;
	for (j = 0; j<ROOMNUM; j++)
	{
		availabl[j] = j + 1;
	}
	available = ROOMNUM;
}

//------------------------------------------------------------------------------
void xy_choose(void)       // Picking a room at random
{
	long rl, s;
	rl = randl(available);
	s = availabl[rl];
	x = (s - 1) % SIDE;
	y = (s - 1) / SIDE;
	availabl[rl] = availabl[available - 1];
	available--;
}

//------------------------------------------------------------------------------
void fresh_unit(void)     // Updating a unit for the next time step
{
	p1 = p->prior;
	p2 = p->next;
	p3 = room_head[!h][y][x]->next;
	room_head[!h][y][x]->next = p;
	p->next = p3;
	p->prior = room_head[!h][y][x];
	if (p3 != room_head[!h][y][x])p3->prior = p;
	p1->next = p2;
	if (p2 != room_head[h][y][x])p2->prior = p1;
	p = p1;
}

//------------------------------------------------------------------------------
int findseq(char seq[], int seqlength, struct rna *p, int m, int n)  // Finding a specific subsequence in a sequence
{
	int flag1, flag2, length, a, b;
	char inf[MAX_RNA_LENGTH];
	for (a = 0; a<MAX_RNA_LENGTH; a++)inf[a] = 0;

	if (m == 1)                 //find the subsequence in which chain
	{
		length = p->length1;
		for (a = 0; a<length; a++)
			inf[a] = p->information[0][a];
	}
	else if (m == 2)
	{
		if (n == 0) //if no nick
		{
			length = p->length2;
			for (a = 0; a<length; a++)
				inf[a] = p->information[1][length - a - 1];
		}
		else if (n == 1) // before nick
		{
			length = p->nick;
			for (a = 0; a<length; a++)
				inf[a] = p->information[1][length - a - 1];
		}
		else if (n == 2) //after nick
		{
			length = p->length2 - p->nick;
			for (a = 0; a<length; a++)
				inf[a] = p->information[1][p->length2 - a - 1];

		}
	}
	else printf("findseq false: m>2");

	flag1 = 0;                                  // Searching for the subsequence
	if (length >= seqlength)
	{
		for (b = 0; length - seqlength - b >= 0; b++)
		{
			flag2 = 0;
			for (a = 0; a<seqlength; a++)
			{
				if (inf[b + a] == seq[a])continue;
				else { flag2 = 1; break; }
			}
			if (flag2 == 0)break;
		}
		if (flag2 == 1)flag1 = 1;
	}
	else flag1 = 1;

	if (flag1 == 0)return(0);   //Yes, the sequence contains the subsequence
	else return(1);   // No, the sequence does not contain the subsequence
}

//------------------------------------------------------------------------------
void inits(void)         // Initialization of the system
{
	int j, m, k;
	seed(SD);

	synlength = 0;
	for (j = 0; nsrseq[j] != 0; j++)
		synlength++;

	contrlength1 = 0;
	for (j = 0; contrseq1[j] != 0; j++)
		contrlength1++;

	contrlength2 = 0;
	for (j = 0; contrseq2[j] != 0; j++)
		contrlength2++;

	contrlength3 = 0;
	for (j = 0; contrseq3[j] != 0; j++)
		contrlength3++;
	///////////////////////////////

	inoculength = 0;
	for (j = 0; inocuseq[j] != 0; j++)
		inoculength++;

	inoculength1 = 0;
	for (j = 0; inocuseq1[j] != 0; j++)
		inoculength1++;

	inoculength2 = 0;
	for (j = 0; inocuseq2[j] != 0; j++)
		inoculength2++;

	inoculength3 = 0;
	for (j = 0; inocuseq3[j] != 0; j++)
		inoculength3++;

	for (m = 0; m<2; m++)
	{
		for (y = 0; y<SIDE; y++)
		{
			for (x = 0; x<SIDE; x++)
			{
				p1 = (struct rna *)malloc(LEN);
				if (!p1) { printf("\tinit1--memeout\n"); exit(0); }
				room_head[m][y][x] = p1;
				p1->next = room_head[m][y][x];
			}
		}
	}
	for (y = 0; y<SIDE; y++)
	{
		for (x = 0; x<SIDE; x++)
		{
			raw_arr[y][x] = 0;
		}
	}
	for (k = 0; k<TOTAL_MATERIAL; k++)  // Initial distribution of raw material
	{
		x = randl(SIDE);
		y = randl(SIDE);
		raw_arr[y][x]++;
	}
}

//------------------------------------------------------------------------------
void inoculate(void)  // Inoculation of RNA species
{
	int k, k1;
	k = INOCUNUM;
	printf("%d", k);

	for (k = 0; k<INOCUNUM; k++) {
		x = randl(SIDE);
		y = randl(SIDE);
		p2 = (struct rna *)malloc(LEN);
		if (!p2) { printf("\t%dform_monomer--memeout\n", k + 1); exit(0); }
		for (k1 = 0; k1<inoculength; k1++) p2->information[0][k1] = inocuseq[k1];
		p2->information[0][k1] = 0;
		p2->information[1][0] = 0;

		p2->length1 = inoculength;
		p2->length2 = 0;
		p2->nick = 0;
		p2->next = room_head[h][y][x]->next;
		if (p2->next != room_head[h][y][x])(p2->next)->prior = p2;
		room_head[h][y][x]->next = p2;
		p2->prior = room_head[h][y][x];
	}

	for (k = 0; k<INOCUNUM1; k++) {
		x = randl(SIDE);
		y = randl(SIDE);
		p2 = (struct rna *)malloc(LEN);
		if (!p2) { printf("\t%dform_monomer--memeout\n", k + 1); exit(0); }
		for (k1 = 0; k1<inoculength1; k1++) p2->information[0][k1] = inocuseq1[k1];
		p2->information[0][k1] = 0;
		p2->information[1][0] = 0;

		p2->length1 = inoculength1;
		p2->length2 = 0;
		p2->nick = 0;
		p2->next = room_head[h][y][x]->next;
		if (p2->next != room_head[h][y][x])(p2->next)->prior = p2;
		room_head[h][y][x]->next = p2;
		p2->prior = room_head[h][y][x];
	}

	for (k = 0; k<INOCUNUM2; k++) {
		x = randl(SIDE);
		y = randl(SIDE);
		p2 = (struct rna *)malloc(LEN);
		if (!p2) { printf("\t%dform_monomer--memeout\n", k + 1); exit(0); }
		for (k1 = 0; k1<inoculength2; k1++) p2->information[0][k1] = inocuseq2[k1];
		p2->information[0][k1] = 0;
		p2->information[1][0] = 0;

		p2->length1 = inoculength2;
		p2->length2 = 0;
		p2->nick = 0;
		p2->next = room_head[h][y][x]->next;
		if (p2->next != room_head[h][y][x])(p2->next)->prior = p2;
		room_head[h][y][x]->next = p2;
		p2->prior = room_head[h][y][x];
	}

	for (k = 0; k<INOCUNUM3; k++) {
		x = randl(SIDE);
		y = randl(SIDE);
		p2 = (struct rna *)malloc(LEN);
		if (!p2) { printf("\t%dform_monomer--memeout\n", k + 1); exit(0); }
		for (k1 = 0; k1<inoculength3; k1++) p2->information[0][k1] = inocuseq3[k1];
		p2->information[0][k1] = 0;
		p2->information[1][0] = 0;

		p2->length1 = inoculength3;
		p2->length2 = 0;
		p2->nick = 0;
		p2->next = room_head[h][y][x]->next;
		if (p2->next != room_head[h][y][x])(p2->next)->prior = p2;
		room_head[h][y][x]->next = p2;
		p2->prior = room_head[h][y][x];
	}
}

//------------------------------------------------------------------------------
void unit_case(void)      // Action of units (molecules) in the system
{
	int a, b, d, j, k, m, n, randnt, raw_bef, nt_turn;
	double f, rtdaddlig, rtdaddphili;

	avail_xy_init();      // Initialization for xy_choose
	for (d = 0; d<ROOMNUM; d++)
	{
		xy_choose();    // Picks a room at random 
		raw_bef = raw_arr[y][x];
		for (k = 0; k<raw_bef; k++)
		{
			randcaser = randl(2);
			switch (randcaser)
			{
				case 0:  // Forming nt
					if (randd()<PNF)
					{
						raw_arr[y][x]--;
						p3 = (struct rna *)malloc(LEN);
						if (!p3) { printf("\t%d form_monomer--memeout\n", k + 1); exit(0); }
						randnt = randl(4) + 1;
						switch (randnt)
						{
							case 1:  p3->information[0][0] = A; break;
							case 2:  p3->information[0][0] = C; break;
							case 3:  p3->information[0][0] = G; break;
							case 4:  p3->information[0][0] = U; break;
							default: printf("form randnt error");
						}
						p3->information[0][1] = 0;
						p3->information[1][0] = 0;

						p3->length1 = 1;
						p3->length2 = 0;
						p3->nick = 0;

						p3->prior = room_head[!h][y][x];
						p3->next = room_head[!h][y][x]->next;
						if (p3->next != room_head[!h][y][x])(p3->next)->prior = p3;
						room_head[!h][y][x]->next = p3;
					}
					break;

				case 1:   // Raw moving
					if (randd()<PMV)
					{
						randcaser1 = randl(4);   // Four possible directions
						switch (randcaser1)
						{
							case 0:
								if (x>0)
								{
									raw_arr[y][x]--;
									raw_arr[y][x - 1]++;
								}
								break;

							case 1:
								if (x<SIDE - 1)
								{
									raw_arr[y][x]--;
									raw_arr[y][x + 1]++;
								}
								break;

							case 2:
								if (y>0)
								{
									raw_arr[y][x]--;
									raw_arr[y - 1][x]++;
								}
								break;

							case 3:
								if (y<SIDE - 1)
								{
									raw_arr[y][x]--;
									raw_arr[y + 1][x]++;
								}
								break;

							default:printf("raw moving error");
						}
					}
					break;

				default:printf("raw case error");
			}
		}


		for (p = room_head[h][y][x]->next; p != room_head[h][y][x]; p = p->next)
		{
			randcase = randl(6);
			switch (randcase)
			{
				case 0:                        // Random ligation of nucleotides and oligoncleotides
					for (p3 = p->next; p3 != p; p3 = p3->next)
					{
						if (p3 == room_head[h][y][x]) { p3 = room_head[h][y][x]->next; if (p3 == p)break; }
						if (p3->length2 == 0)
						{
							if (randd()<PRL / p3->length1)
							{
								if (p->length1 + p3->length1>MAX_RNA_LENGTH - 1)
								{
									over_max_len++; continue;
								}

								for (a = 0; a<p3->length1; a++)
									p->information[0][a + p->length1] = p3->information[0][a];
								p->information[0][p->length1 + p3->length1] = 0;
								p->length1 = p->length1 + p3->length1;

								(p3->prior)->next = p3->next;
								if (p3->next != room_head[h][y][x])(p3->next)->prior = p3->prior;
								free(p3);

								break;
							}
						}
					}
					fresh_unit();
					break;

				case 1:            // Decay and degradation 
					if (p->length1 == 1)  // Decay of nucleotide
					{
						if (p->length2 == 0 && randd()<PND)
						{
							raw_arr[y][x]++;
							(p->prior)->next = p->next;
							if (p->next != room_head[h][y][x])(p->next)->prior = p->prior;
							p3 = p;
							p = p->prior;
							free(p3); break;
						}
					}
					else                  //Degradation of chain
					{
						f = PBB;
						for (j = p->length1; j>1; j--)
						{
							if (j <= p->length2)   // Falling into double chain region
							{
								if (p->nick == 0)
								{
									m = j - 1;
									n = p->length2 - j + 1;
									k = (m<n) ? m : n;
									f = PBB * k*PBB*k;
								}
								else
								{
									if (j == p->nick + 1)
									{
										m = j - 1;
										n = p->length2 - j + 1;
										k = (m<n) ? m : n;
										f = PBB * k;
									}
									else if (j>p->nick + 1)
									{
										m = j - p->nick - 1;
										n = p->length2 - j + 1;
										k = (m<n) ? m : n;
										f = PBB * k*PBB*k;
									}
									else
									{
										m = j - 1;
										n = p->nick - j + 1;
										k = (m<n) ? m : n;
										f = PBB * k*PBB*k;
									}
								}
							}

							if (randd()<f)
							{
								p3 = (struct rna *)malloc(LEN);
								if (!p3) { printf("\t%ddeg--memeout\n", i); exit(0); }

								for (b = 0; b<p->length1 - j + 1; b++)
									p3->information[0][b] = p->information[0][b + j - 1];
								p3->information[0][p->length1 - j + 1] = 0;
								p->information[0][j - 1] = 0;
								p3->length1 = p->length1 - j + 1;
								p->length1 = j - 1;

								if (p->length2>j - 1)
								{
									for (b = 0; b<p->length2 - j + 1; b++)
										p3->information[1][b] = p->information[1][b + j - 1];
									p3->information[1][p->length2 - j + 1] = 0;
									p->information[1][j - 1] = 0;
									p3->length2 = p->length2 - j + 1;
									p->length2 = j - 1;
								}
								else
								{
									p3->information[1][0] = 0;
									p3->length2 = 0;
								}

								if (p->nick>j - 1) { p3->nick = p->nick - j + 1; p->nick = 0; }
								else if (p->nick == j - 1) { p3->nick = 0; p->nick = 0; }
								else p3->nick = 0;

								p3->prior = room_head[!h][y][x];
								p3->next = room_head[!h][y][x]->next;
								if (p3->next != room_head[!h][y][x])(p3->next)->prior = p3;
								room_head[!h][y][x]->next = p3;
								break;
							}
						}
					}
					fresh_unit();
					break;

				case 2:                         // Template-directed addition
					if (p->nick == 0)                      // Template-directed attraction of substrates
					{
						for (p3 = p->next; p3 != p; p3 = p3->next)
						{
							if (p3 == room_head[h][y][x]) { p3 = room_head[h][y][x]->next; if (p3 == p)break; }
							if (p3->length2 == 0)
							{
								if (p3->length1 <= p->length1 - p->length2)
								{
									for (flag = 0, b = 0; b<p3->length1; b++)
									{
										if ((p3->information[0][p3->length1 - 1 - b] + p->information[0][p->length2 + b]) == 5)continue;
										else if (randd()<PFP)continue;
										else { flag = 1; break; }
									}
									if (flag == 0)
									{
										rtdaddphili = randd();
										if (rtdaddphili<PAT)
										{
											for (a = 0; a<p3->length1; a++)
												p->information[1][p->length2 + a] = p3->information[0][p3->length1 - 1 - a];
											p->information[1][p->length2 + p3->length1] = 0;
											if (p->length2 != 0)p->nick = p->length2;
											p->length2 = p->length2 + p3->length1;

											(p3->prior)->next = p3->next;
											if (p3->next != room_head[h][y][x])(p3->next)->prior = p3->prior;
											free(p3);
											break;
										}
									}
								}
							}
						}
					}
					else                     // Template-directed ligation
					{
						rtdaddlig = randd();
						if (rtdaddlig<PLT)
						{
							p->nick = 0;
						}
					}
					fresh_unit();
					break;

				case 3:                           // Separation
					if (p->length2 != 0)    // Separation of double chain  
					{
						if (randd()<pow(PSBP, p->length2 - p->nick))
						{
							p3 = (struct rna *)malloc(LEN);
							if (!p3) { printf("\t%dsep--memeout\n", i); exit(0); }
							for (b = 0; b<p->length2 - p->nick; b++)
								p3->information[0][b] = p->information[1][p->length2 - 1 - b];
							p->information[1][p->nick] = 0;

							p3->information[0][p->length2 - p->nick] = 0;
							p3->information[1][0] = 0;

							p3->length1 = p->length2 - p->nick;
							p->length2 = p->nick;
							p->nick = 0;
							p3->length2 = 0;
							p3->nick = 0;

							p3->prior = room_head[!h][y][x];
							p3->next = room_head[!h][y][x]->next;
							if (p3->next != room_head[!h][y][x])(p3->next)->prior = p3;
							room_head[!h][y][x]->next = p3;
						}
					}
					fresh_unit();
					break;

					//-------------------------------------------------------------------------------------
				case 4:
					if (p->length2 == 0)
					{
						flag = findseq(nsrseq, synlength, p, 1, 0);
						if (flag == 0)    // NSR catalyzing the synthesis of nucleotides. 
						{
							nt_turn = TNSS;
							raw_bef = raw_arr[y][x];
							for (k = 0; k<raw_bef; k++)
							{
								if (nt_turn <= 0)break;
								nt_turn--;
								if (randd()<PNFR)
								{
									raw_arr[y][x]--;

									p3 = (struct rna *)malloc(LEN);
									if (!p3) { printf("\t%dsyn form_monomer--memeout\n", k + 1); exit(0); }
									randnt = randl(4) + 1;
									switch (randnt)
									{
										case 1:  p3->information[0][0] = A; break;
										case 2:  p3->information[0][0] = C; break;
										case 3:  p3->information[0][0] = G; break;
										case 4:  p3->information[0][0] = U; break;
										default: printf("syn randnt error");
									}
									p3->information[0][1] = 0;
									p3->information[1][0] = 0;

									p3->length1 = 1;
									p3->length2 = 0;
									p3->nick = 0;

									p3->prior = room_head[!h][y][x];
									p3->next = room_head[!h][y][x]->next;
									if (p3->next != room_head[!h][y][x])(p3->next)->prior = p3;
									room_head[!h][y][x]->next = p3;
								}
							}
						}
					}
					fresh_unit();
					break;

				case 5:            // Moving to another adjacent room
					if (randd()*FDMOV<PMN)
					{
						randcase1 = randl(4);   // Four possible directions
						switch (randcase1)
						{
							case 0:
								if (x>0)
								{
									p1 = p->prior;
									p2 = p->next;

									p3 = room_head[!h][y][x - 1]->next;
									room_head[!h][y][x - 1]->next = p;
									p->next = p3;
									p->prior = room_head[!h][y][x - 1];
									if (p3 != room_head[!h][y][x - 1])p3->prior = p;

									p1->next = p2;
									if (p2 != room_head[h][y][x])p2->prior = p1;
									p = p1;
								}
								else fresh_unit();
								break;

							case 1:
								if (x<SIDE - 1)
								{
									p1 = p->prior;
									p2 = p->next;

									p3 = room_head[!h][y][x + 1]->next;
									room_head[!h][y][x + 1]->next = p;
									p->next = p3;
									p->prior = room_head[!h][y][x + 1];
									if (p3 != room_head[!h][y][x + 1])p3->prior = p;

									p1->next = p2;
									if (p2 != room_head[h][y][x])p2->prior = p1;
									p = p1;
								}
								else fresh_unit();
								break;

							case 2:
								if (y>0)
								{
									p1 = p->prior;
									p2 = p->next;

									p3 = room_head[!h][y - 1][x]->next;
									room_head[!h][y - 1][x]->next = p;
									p->next = p3;
									p->prior = room_head[!h][y - 1][x];
									if (p3 != room_head[!h][y - 1][x])p3->prior = p;

									p1->next = p2;
									if (p2 != room_head[h][y][x])p2->prior = p1;
									p = p1;
								}
								else fresh_unit();
								break;


							case 3:
								if (y<SIDE - 1)
								{
									p1 = p->prior;
									p2 = p->next;

									p3 = room_head[!h][y + 1][x]->next;
									room_head[!h][y + 1][x]->next = p;
									p->next = p3;
									p->prior = room_head[!h][y + 1][x];
									if (p3 != room_head[!h][y + 1][x])p3->prior = p;

									p1->next = p2;
									if (p2 != room_head[h][y][x])p2->prior = p1;
									p = p1;
								}
								else fresh_unit();
								break;

							default:printf("rna moving error");
						}
					}
					else fresh_unit();
					break;

				default: printf("rna case error");
			}
		}
	}
}

//------------------------------------------------------------------------------
int record(void)               // Data recording
{
	recstep[g] = i;

	NSR_num[g] = 0;
	contr_num1[g] = 0;
	contr_num2[g] = 0;
	contr_num3[g] = 0;

	total_mat_num[g] = 0;
	unit_num[g] = 0;
	raw_num[g] = 0;
	for (y = 0; y<SIDE; y++)
	{
		for (x = 0; x<SIDE; x++)
		{
			raw_num[g] += raw_arr[y][x];
			for (p = room_head[h][y][x]->next; p != room_head[h][y][x]; p = p->next)
			{
				unit_num[g]++;
				total_mat_num[g] += p->length1 + p->length2;
				flag1 = findseq(nsrseq, synlength, p, 1, 0);
				if (flag1 == 0) NSR_num[g]++; //synthetase sequence in chain1
				flag1 = findseq(contrseq1, contrlength1, p, 1, 0);
				if (flag1 == 0) contr_num1[g]++; //
				flag1 = findseq(contrseq2, contrlength2, p, 1, 0);
				if (flag1 == 0) contr_num2[g]++; //
				flag1 = findseq(contrseq3, contrlength3, p, 1, 0);
				if (flag1 == 0) contr_num3[g]++; //
			}
		}
	}
	total_mat_num[g] += raw_num[g];
	g++;
	return(0);
}

//------------------------------------------------------------------------------
void save_result(int aa, int bb)    // Data saving
{
	//FILE *fp1, *fp2, *fp3, *fp4, *fp5;

	FILE *fptxt1;
	errno_t err;
	err = fopen_s(&fptxt1, "file1.txt", "at");
	if (err != 0) { printf("cannot open file");  exit(-1); }

	int chnumns[MAX_RNA_LENGTH], chnums[MAX_RNA_LENGTH], si;
	int ch[SIDE][SIDE], ch_syn[SIDE][SIDE];

	printf("chain_num(NSR_num):\n");
	for (y = 0; y<SIDE; y++)
	{
		for (x = 0; x<SIDE; x++)
		{
			ch[y][x] = 0;
			ch_syn[y][x] = 0;
			for (p = room_head[h][y][x]->next; p != room_head[h][y][x]; p = p->next)
			{
				ch[y][x]++;
				flag1 = findseq(nsrseq, synlength, p, 1, 0);
				if (flag1 == 0)ch_syn[y][x]++;
			}
			printf("%d(%d)\t", ch[y][x], ch_syn[y][x]);
		}
		printf("\n");
	}

	for (si = 0; si<MAX_RNA_LENGTH; si++)
	{
		chnumns[si] = 0;
		chnums[si] = 0;
	}

	for (y = 0; y<SIDE; y++)
	{
		for (x = 0; x<SIDE; x++)
		{
			for (p = room_head[h][y][x]->next; p != room_head[h][y][x]; p = p->next)
			{
				flag1 = findseq(nsrseq, synlength, p, 1, 0);
				if (flag1 == 0)chnums[p->length1 - 1]++;
				else chnumns[p->length1 - 1]++;
			}
		}
	}

	printf("over_max_length = %d times\n", over_max_len);
	printf("\nsyn,contr1,contr2,contr3=\n");

	for (g = 0; g<(STEPNUM - STAREC) / RECINT + 1; g++)
	{
		gi = g * RECINT + STAREC;
		printf("%d,%d,%d,%d(%d,%d,u%d,r%d)\t", (int)NSR_num[g], (int)contr_num1[g], (int)contr_num2[g], (int)contr_num3[g], gi, (int)total_mat_num[g], (int)unit_num[g], (int)raw_num[g]);
		fprintf(fptxt1, "%d, ", (int)NSR_num[g]);

	}
	if(bb==0) fprintf(fptxt1,"--- aaa=%d\n",aa);
	else fprintf(fptxt1, "\n");

	printf("\nend\n");

	fclose(fptxt1);
}

//------------------------------------------------------------------------------
void freepool(void)        // Memory releasing  
{
	int m;
	for (m = 0; m<2; m++)
	{
		for (y = 0; y<SIDE; y++)
		{
			for (x = 0; x<SIDE; x++)
			{
				while (1)
				{
					if (room_head[m][y][x]->next != room_head[m][y][x])
					{
						p = room_head[m][y][x]->next;
						room_head[m][y][x]->next = p->next;
						free(p);
					}
					else break;
				}
				free(room_head[m][y][x]);
			}
		}
	}
}


double max_fabs(double a, double b,	double c, double d,	double e, double f,	double g, double h)  // Finding out the greatest absolute value
{
	a=fabs(a); b=fabs(b); c=fabs(c); d=fabs(d); e=fabs(e); f=fabs(f); g=fabs(g); h=fabs(h); 

	double mmax = 0;
	if (a > mmax)
		mmax = a;
	if (b > mmax)
		mmax = b;
	if (c > mmax)
		mmax = c;
	if (d > mmax)
		mmax = d;
	if (e > mmax)
		mmax = e;
	if (f > mmax)
		mmax = f;
	if (g > mmax)
		mmax = g;
	if (h > mmax)
		mmax = h;
	return mmax;
}


//------------------------------------------------------------------------------ 
int main()
{
	double max;
	FILE *fptxt2,*fptxt3;
	errno_t err2,err3;

	//*** Setting the starting point of machine-learning
	PNF = 0.004;         
	PNFR = 0.02;         
	PND = 0.001;         
	PRL = 0.00002;        
	PBB = 0.00001;       
	PAT = 0.5;    
	PFP = 0.1;    
	PMV = 0.001;  
	//**************************************************

	int aaa, bbb;
	for(aaa=0;aaa<=LEARNING_ROUND;aaa++)      //*** Learning-cycle
	{
		for (bbb = 0; bbb <= 8; bbb++)       //*** Testing the influence of the parameters on the objective function (NSR number)
		{
			switch (bbb)
			{
				case 1:PARA[0][bbb][aaa] = PNF;
					if (aaa != 0)
					{
						if(deltaPARA[bbb][aaa-1]==0) PNF *= (1+learning_rate);
						else PNF = PNF * (1 + learning_rate * ((PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]) / fabs(PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]))  *  (deltaPARA[bbb][aaa-1] / fabs(deltaPARA[bbb][aaa-1])));
					}
					else PNF *= (1+learning_rate);
					PARA[1][bbb][aaa] = PNF;
					break;

				case 2:PARA[0][bbb][aaa] = PNFR;
					if (aaa != 0)
					{
						if(deltaPARA[bbb][aaa-1]==0) PNFR *= (1+learning_rate);
						else PNFR = PNFR * (1 + learning_rate * ((PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]) / fabs(PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]))*(deltaPARA[bbb][aaa-1] / fabs(deltaPARA[bbb][aaa-1])));
					}
					else PNFR *= (1+learning_rate);
					PARA[1][bbb][aaa] = PNFR;
					break;

				case 3:PARA[0][bbb][aaa] = PND;
					if (aaa != 0)
					{
						if(deltaPARA[bbb][aaa-1]==0) PND *= (1+learning_rate);
						else PND = PND * (1 + learning_rate * ((PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]) / fabs(PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]))*(deltaPARA[bbb][aaa-1] / fabs(deltaPARA[bbb][aaa-1])));
					}
					else PND *= (1+learning_rate);
					PARA[1][bbb][aaa] = PND;
					break;

				case 4:PARA[0][bbb][aaa] = PRL;
					if (aaa != 0)
					{
						if(deltaPARA[bbb][aaa-1]==0) PRL *= (1+learning_rate);
						else PRL = PRL * (1 + learning_rate * ((PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]) / fabs(PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]))*(deltaPARA[bbb][aaa-1] / fabs(deltaPARA[bbb][aaa-1])));
					}
					else PRL *= (1+learning_rate);
					PARA[1][bbb][aaa] = PRL;
					break;

				case 5:PARA[0][bbb][aaa] = PBB;
					if (aaa != 0)
					{
						if(deltaPARA[bbb][aaa-1]==0) PBB *= (1+learning_rate);
						else PBB = PBB * (1 + learning_rate * ((PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]) / fabs(PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]))*(deltaPARA[bbb][aaa-1] / fabs(deltaPARA[bbb][aaa-1])));
					}
					else PBB *= (1+learning_rate);
					PARA[1][bbb][aaa] = PBB;
					break;

				case 6:PARA[0][bbb][aaa] = PAT;
					if (aaa != 0)
					{
						if(deltaPARA[bbb][aaa-1]==0) PAT *= (1+learning_rate);
						else PAT = PAT * (1 + learning_rate * ((PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]) / fabs(PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]))*(deltaPARA[bbb][aaa-1] / fabs(deltaPARA[bbb][aaa-1])));
					}
					else PAT *= (1+learning_rate);
					PARA[1][bbb][aaa] = PAT;
					break;

				case 7:PARA[0][bbb][aaa] = PFP;
					if (aaa != 0)
					{
						if(deltaPARA[bbb][aaa-1]==0) PFP *= (1+learning_rate);
						else PFP = PFP * (1 + learning_rate * ((PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]) / fabs(PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]))*(deltaPARA[bbb][aaa-1] / fabs(deltaPARA[bbb][aaa-1])));
					}
					else PFP *= (1+learning_rate);
					PARA[1][bbb][aaa] = PFP;
					break;

				case 8:PARA[0][bbb][aaa] = PMV;
					if (aaa != 0)
					{
						if(deltaPARA[bbb][aaa-1]==0) PMV *= (1+learning_rate);
						else PMV = PMV * (1 + learning_rate * ((PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]) / fabs(PARA[1][bbb][aaa - 1] - PARA[0][bbb][aaa - 1]))*(deltaPARA[bbb][aaa-1] / fabs(deltaPARA[bbb][aaa-1])));
					}
					else PMV *= (1+learning_rate);
					PARA[1][bbb][aaa] = PMV;
					break;

				default:
					break;
			}
			nd = 0;
			h = 0;
			g = 0;
			inits();        // Initialization of the system
			for (i = 0; i <= STEPNUM; i++)      // Monte-Carlo cycle
			{
				if (i == INOCUSTEP)inoculate();
				if (i >= STAREC && i%RECINT == 0)
				{
					record();
				}
				printf("aaa=%d, bbb=%d, go%ld\n",aaa,bbb, i);
				unit_case();
				h = !h;	
			}
			save_result(aaa,bbb);
			NSR_num_PARA[bbb][aaa] = NSR_num[(STEPNUM - STAREC) / RECINT];
			freepool();

			switch (bbb)   //*** Regaining a parameter's previous value for testing other parameters in this learning round
			{
				case 1:		PNF  = PARA[0][bbb][aaa];	break; 
				case 2:		PNFR = PARA[0][bbb][aaa];   break;
				case 3:		PND  = PARA[0][bbb][aaa];	break;
				case 4:		PRL  = PARA[0][bbb][aaa];	break;
				case 5:		PBB  = PARA[0][bbb][aaa]; 	break;
				case 6:		PAT  = PARA[0][bbb][aaa];	break;
				case 7:		PFP  = PARA[0][bbb][aaa];   break;
				case 8:		PMV  = PARA[0][bbb][aaa]; 	break;
				default:	break;
			}
		}

		//*** Caculating the change of the objective function (NSR number) brought by the change of a parameter
		deltaPARA[1][aaa] = NSR_num_PARA[1][aaa] - NSR_num_PARA[0][aaa];
		deltaPARA[2][aaa] = NSR_num_PARA[2][aaa] - NSR_num_PARA[0][aaa];
		deltaPARA[3][aaa] = NSR_num_PARA[3][aaa] - NSR_num_PARA[0][aaa];
		deltaPARA[4][aaa] = NSR_num_PARA[4][aaa] - NSR_num_PARA[0][aaa];
		deltaPARA[5][aaa] = NSR_num_PARA[5][aaa] - NSR_num_PARA[0][aaa];
		deltaPARA[6][aaa] = NSR_num_PARA[6][aaa] - NSR_num_PARA[0][aaa];
		deltaPARA[7][aaa] = NSR_num_PARA[7][aaa] - NSR_num_PARA[0][aaa];
		deltaPARA[8][aaa] = NSR_num_PARA[8][aaa] - NSR_num_PARA[0][aaa];
		//*****************************************************************************************************

		err2 = fopen_s(&fptxt2, "file2.txt", "at");
		if (err2 != 0) { printf("cannot open file2");  exit(-1); }
		err3 = fopen_s(&fptxt3, "file3.txt", "at");
		if (err3 != 0) { printf("cannot open file3");  exit(-1); }

		if(aaa<10)fprintf(fptxt2, "aaa=%d,  PNF=%e, PNFR=%e, PND=%e, PRL=%e, PBB=%e, PAT=%e, PFP=%e, PMV=%e, syn=%d\n",aaa,PNF, PNFR, PND, PRL, PBB, PAT, PFP, PMV, (int)NSR_num_PARA[0][aaa]);
		else fprintf(fptxt2, "aaa=%d, PNF=%e, PNFR=%e, PND=%e, PRL=%e, PBB=%e, PAT=%e, PFP=%e, PMV=%e, syn=%d\n",aaa,PNF, PNFR, PND, PRL, PBB, PAT, PFP, PMV, (int)NSR_num_PARA[0][aaa]);
		max = max_fabs(deltaPARA[1][aaa], deltaPARA[2][aaa], deltaPARA[3][aaa], deltaPARA[4][aaa], deltaPARA[5][aaa], deltaPARA[6][aaa], deltaPARA[7][aaa],deltaPARA[8][aaa]);
		//*** Finding out the greatest change of the objective function (NSR number) brought by the change of a parameter

		fprintf(fptxt3, "%d,",(int)NSR_num_PARA[0][aaa]);

		//*** Preparing parameters for next round of learning - according to the greatest gradient
		PNF  = PARA[0][1][aaa] * (1 + learning_rate * ((PARA[1][1][aaa] - PARA[0][1][aaa]) / fabs(PARA[1][1][aaa] - PARA[0][1][aaa]))  *  (deltaPARA[1][aaa] / max)); 
		PNFR = PARA[0][2][aaa] * (1 + learning_rate * ((PARA[1][2][aaa] - PARA[0][2][aaa]) / fabs(PARA[1][2][aaa] - PARA[0][2][aaa]))  *  (deltaPARA[2][aaa] / max));
		PND  = PARA[0][3][aaa] * (1 + learning_rate * ((PARA[1][3][aaa] - PARA[0][3][aaa]) / fabs(PARA[1][3][aaa] - PARA[0][3][aaa]))  *  (deltaPARA[3][aaa] / max));
		PRL  = PARA[0][4][aaa] * (1 + learning_rate * ((PARA[1][4][aaa] - PARA[0][4][aaa]) / fabs(PARA[1][4][aaa] - PARA[0][4][aaa]))  *  (deltaPARA[4][aaa] / max));
		PBB  = PARA[0][5][aaa] * (1 + learning_rate * ((PARA[1][5][aaa] - PARA[0][5][aaa]) / fabs(PARA[1][5][aaa] - PARA[0][5][aaa]))  *  (deltaPARA[5][aaa] / max));
		PAT  = PARA[0][6][aaa] * (1 + learning_rate * ((PARA[1][6][aaa] - PARA[0][6][aaa]) / fabs(PARA[1][6][aaa] - PARA[0][6][aaa]))  *  (deltaPARA[6][aaa] / max));
		PFP  = PARA[0][7][aaa] * (1 + learning_rate * ((PARA[1][7][aaa] - PARA[0][7][aaa]) / fabs(PARA[1][7][aaa] - PARA[0][7][aaa]))  *  (deltaPARA[7][aaa] / max));
		PMV  = PARA[0][8][aaa] * (1 + learning_rate * ((PARA[1][8][aaa] - PARA[0][8][aaa]) / fabs(PARA[1][8][aaa] - PARA[0][8][aaa]))  *  (deltaPARA[8][aaa] / max));
		//*******************************************************************************

		fclose(fptxt2);	
		fclose(fptxt3);	
    }
	return 0;
}
//========================================================  End of the program

