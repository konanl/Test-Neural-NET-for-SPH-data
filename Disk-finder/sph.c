#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct io_header_1
{
  int npart[6];
  double mass[6];
  double time;
  double redshift;
  int flag_sfr;
  int flag_feedback;
  int npartTotal[6];
  int flag_cooling;
  int num_files;
  double BoxSize;
  double Omega0;
  double OmegaLambda;
  double HubbleParam;
  int flag_stellarage;
  int flag_metals;
  unsigned int npartTotalHighWord[6];
  int  flag_entropy_instead_u;
  char fill[60];
  //char fill[256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8];	/* fills to 256 Bytes */
} header1;


int NumPart, Ngas;

struct particle_data
{
  float Pos[3];
  float Vel[3];
  float Mass;
  int Type;

  //float Rho, U, Temp, Ne;
  float Rho, U, Temp, Hsml;
} *P;

int *Id;

double Time, Redshift;

/* Here we load a snapshot file. It can be distributed
 * onto several files (for files>1).
 * The particles are brought back into the order
 * implied by their ID's.
 * A unit conversion routine is called to do unit
 * conversion, and to evaluate the gas temperature.
 */
int allocate_memory(void);
int do_what_you_want(char *filename);
int unit_conversion(void);
int load_snapshot(char *fname, int files);
int reordering(void);
int main(int argc, char **argv)
{
	if(argc!=7)
	{
		printf("Parameter Count Error!\n");
		return 1;
	}
	char pt1[8]={'\0'};
	char pt2[8]={'\0'};
	char pt3[8]={'\0'};
	//char pt4[8]={'\0'};
	//char outfile[256]={'\0'};

	 char path[200]={'\0'};
	 char input_fname[256]={'\0'};
	 char basename[256]={'\0'};
	  int snapshot_number, files;
	  snapshot_number=0;
	  files=0;
	sprintf(pt1,"%s",argv[1]);
	sprintf(pt2,"%s",argv[3]);
	sprintf(pt3,"%s",argv[5]);
	//sprintf(pt4,"%s",argv[7]);
	//判断第一个参数类型
	if(strcmp(pt1,"-path")==0)
	{
		sprintf(path,"%s",argv[2]);
	}
	else if(strcmp(pt1,"-dump")==0)
	{
		snapshot_number=atoi(argv[2]);
	}
	else if(strcmp(pt1,"-count")==0)
	{
		files=atoi(argv[2]);
	}


	//判断第二个参数类型
	if(strcmp(pt2,"-path")==0)
	{
		sprintf(path,"%s",argv[4]);
	}
	else if(strcmp(pt2,"-dump")==0)
	{
		snapshot_number=atoi(argv[4]);
	}
	else if(strcmp(pt2,"-count")==0)
	{
		files=atoi(argv[4]);
	}
	/*else if(strcmp(pt2,"-out")==0)
	{
		sprintf(outfile,"%s",argv[4]);
	}*/

	//判断第三个参数类型
	if(strcmp(pt3,"-path")==0)
	{
		sprintf(path,"%s",argv[6]);
	}
	else if(strcmp(pt3,"-dump")==0)
	{
		snapshot_number=atoi(argv[6]);
	}
	else if(strcmp(pt3,"-count")==0)
	{
		files=atoi(argv[6]);
	}

	if(strlen(path)==0||files==0)//参数都是正确的
	{
		//printf("Path:%s\nsnapshot_number:%d\nfiles:%d\noutfile:%s\n",path,snapshot_number,files,outfile);
		printf("Parameter Error!\n");
		return 1;
	}

  sprintf(basename, "snapshot");
  sprintf(input_fname, "%s/%s_%03d", path, basename, snapshot_number);
  load_snapshot(input_fname, files);


  //reordering();			/* call this routine only if your ID's are set properly */

  unit_conversion();		/* optional stuff */

  do_what_you_want(input_fname);
  return 0;
}

/* here the particle data is at your disposal
 */
int do_what_you_want(char *filename)
{
	char head_fname[256]={'\0'};//头文件数据
	char id_fname[256]={'\0'};  //id数据
	char pos_fname[256]={'\0'}; //位置数据
	char vel_fname[256]={'\0'}; //速度数据
	sprintf(head_fname, "%s-Head.txt", filename);
	sprintf(id_fname, "%s-Id.txt", filename);
	sprintf(pos_fname, "%s-Pos.txt", filename);
	sprintf(vel_fname, "%s-Vel.txt", filename);

	//解析头文件数据
	FILE *head=fopen(head_fname,"w");
	if(head==NULL)
	{
		printf("Open Head File Failed!\n");
		return -1;
	}
	fprintf(head,"\n");
	fprintf(head,"Npart : %d %d %d %d %d %d \n",header1.npart[0],header1.npart[1],header1.npart[2],header1.npart[3],header1.npart[4],header1.npart[5]);
	fprintf(head,"Massarr : %f %f %f %f %f %f \n",header1.mass[0],header1.mass[1],header1.mass[2],header1.mass[3],header1.mass[4],header1.mass[5]);
	fprintf(head,"Time : %f \n",header1.time);
	fprintf(head,"Redshift : %f \n",header1.redshift);
	fprintf(head,"FlagSfr : %d \n",header1.flag_sfr);
	fprintf(head,"FlagFeedback : %d \n",header1.flag_feedback);
	fprintf(head,"Nall : %d %d %d %d %d %d \n",header1.npartTotal[0],header1.npartTotal[1],header1.npartTotal[2],header1.npartTotal[3],header1.npartTotal[4],header1.npartTotal[5]);
	fprintf(head,"FlagCooling : %d \n",header1.flag_cooling);
	fprintf(head,"NumFiles : %d \n",header1.num_files);
	fprintf(head,"BoxSize : %f \n",header1.BoxSize);
	fprintf(head,"Omega0 : %f \n",header1.Omega0);
	fprintf(head,"OmegaLambda : %f \n",header1.OmegaLambda);
	fprintf(head,"HubbleParam : %f \n",header1.HubbleParam);
	fprintf(head,"FlagAge : %d \n",header1.flag_stellarage);
	fprintf(head,"FlagMetals : %d \n",header1.flag_metals);
	fprintf(head,"NallHW : %d %d %d %d %d %d \n",header1.npartTotalHighWord[0],header1.npartTotalHighWord[1],header1.npartTotalHighWord[2],header1.npartTotalHighWord[3],header1.npartTotalHighWord[4],header1.npartTotalHighWord[5]);
	fprintf(head,"flag_entr_ics : %d ",header1.flag_entropy_instead_u);
	fclose(head);
	head=NULL;

	//解析ID数据
	FILE *idfile=fopen(id_fname,"w");
	if(idfile==NULL)
	{
		printf("Open id File Failed!\n");
		return -1;
	}
	int i=0;
	for(i = 1; i <= NumPart; i++)
	{
		fprintf(idfile,"\nid[%d] : %d",i-1,Id[i]);
	}
	fclose(idfile);

	//解析Pos数据
	FILE *posfile=fopen(pos_fname,"w");
	if(posfile==NULL)
	{
		printf("Open Pos File Failed!\n");
		return -1;
	}
	int j=0;
	for(j = 1; j <= NumPart; j++)
	{
		fprintf(posfile,"\npos[%d][3] : %f %f %f",j-1,P[j].Pos[0],P[j].Pos[1],P[j].Pos[2]);
	}
	fclose(posfile);

	//解析Vel数据
	FILE *velfile=fopen(vel_fname,"w");
	if(velfile==NULL)
	{
		printf("Open Vel File Failed!\n");
		return -1;
	}
	int k=0;
	for(k = 1; k <= NumPart; k++)
	{
		fprintf(velfile,"\nvel[%d][3] : %f %f %f",k-1,P[k].Vel[0],P[k].Vel[1],P[k].Vel[2]);
		//fprintf(velfile,"\nvel[%d][3] : %f %f %f %f %f %f %f",k-1,P[k].Vel[0],P[k].Vel[1],P[k].Vel[2],P[k].Mass,P[k].U,P[k].Rho,P[k].Hsml);
		//fprintf(velfile,"\nvel[%d][3] : %f %f %f  mass:%f   U:%f    rho:%f   hsml:%f",k-1,P[k].Vel[0],P[k].Vel[1],P[k].Vel[2],P[k].Mass,P[k].U,P[k].Rho,P[k].Hsml);
	}
	fclose(velfile);

	return 0;
}

/* this template shows how one may convert from Gadget's units
 * to cgs units.
 * In this example, the temperate of the gas is computed.
 * (assuming that the electron density in units of the hydrogen density
 * was computed by the code. This is done if cooling is enabled.)
 */
int unit_conversion(void)
{
  double GRAVITY, BOLTZMANN, PROTONMASS;
  double UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
  double UnitTime_in_s, UnitDensity_in_cgs, UnitPressure_in_cgs, UnitEnergy_in_cgs;
  double G, Xh, HubbleParam;

  int i;
  double MeanWeight, u, gamma;

  /* physical constants in cgs units */
  GRAVITY = 6.672e-8;
  BOLTZMANN = 1.3806e-16;
  PROTONMASS = 1.6726e-24;

  /* internal unit system of the code */
  UnitLength_in_cm = 3.085678e21;	/*  code length unit in cm/h */
  UnitMass_in_g = 1.989e43;	/*  code mass unit in g/h */
  UnitVelocity_in_cm_per_s = 1.0e5;

  UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s;
  UnitDensity_in_cgs = UnitMass_in_g / pow(UnitLength_in_cm, 3);
  UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / pow(UnitTime_in_s, 2);
  UnitEnergy_in_cgs = UnitMass_in_g * pow(UnitLength_in_cm, 2) / pow(UnitTime_in_s, 2);

  G = GRAVITY / pow(UnitLength_in_cm, 3) * UnitMass_in_g * pow(UnitTime_in_s, 2);


  Xh = 0.76;			/* mass fraction of hydrogen */
  HubbleParam = 0.65;


  for(i = 1; i <= NumPart; i++)
    {
      if(P[i].Type == 0)	/* gas particle */
	{
	  //MeanWeight = 4.0 / (3 * Xh + 1 + 4 * Xh * P[i].Ne) * PROTONMASS;
    MeanWeight = 4.0 / (3 * Xh + 1 + 4 * Xh * P[i].Hsml) * PROTONMASS;
	  /* convert internal energy to cgs units */

	  u = P[i].U * UnitEnergy_in_cgs / UnitMass_in_g;

	  gamma = 5.0 / 3;

	  /* get temperature in Kelvin */

	  P[i].Temp = MeanWeight / BOLTZMANN * (gamma - 1) * u;
	}
    }
  return 0;
}

/* this routine loads particle data from Gadget's default
 * binary file format. (A snapshot may be distributed
 * into multiple files.
 */
int load_snapshot(char *fname, int files)
{
  FILE *fd;
  char buf[200];
  int i, k, dummy, ntot_withmasses;
  int  n, pc, pc_new, pc_sph;

#define SKIP fread(&dummy, sizeof(dummy), 1, fd);

  for(i = 0, pc = 1; i < files; i++, pc = pc_new)
    {
      if(files > 1)
	sprintf(buf, "%s.%d", fname, i);
      else
	sprintf(buf, "%s", fname);

      if(!(fd = fopen(buf, "r")))
	{
	  printf("can't open file `%s`\n", buf);
	  exit(0);
	}

      printf("reading `%s' ...\n", buf);
      fflush(stdout);

      fread(&dummy, sizeof(dummy), 1, fd);
      fread(&header1, sizeof(header1), 1, fd);
      fread(&dummy, sizeof(dummy), 1, fd);

      if(files == 1)
	{
	  for(k = 0, NumPart = 0, ntot_withmasses = 0; k < 6; k++)
	    NumPart += header1.npart[k];
	  Ngas = header1.npart[0];
	}
      else
	{
	  for(k = 0, NumPart = 0, ntot_withmasses = 0; k < 6; k++)
	    NumPart += header1.npartTotal[k];
	  Ngas = header1.npartTotal[0];
	}

      for(k = 0, ntot_withmasses = 0; k < 6; k++)
	{
	  if(header1.mass[k] == 0)
	    ntot_withmasses += header1.npart[k];
	}

      if(i == 0)
	allocate_memory();

      SKIP;
      for(k = 0, pc_new = pc; k < 6; k++)
	{
	  for(n = 0; n < header1.npart[k]; n++)
	    {
	      fread(&P[pc_new].Pos[0], sizeof(float), 3, fd);
	      pc_new++;
	    }
	}
      SKIP;

      SKIP;
      for(k = 0, pc_new = pc; k < 6; k++)
	{
	  for(n = 0; n < header1.npart[k]; n++)
	    {
	      fread(&P[pc_new].Vel[0], sizeof(float), 3, fd);
	      pc_new++;
	    }
	}
      SKIP;


      SKIP;
      for(k = 0, pc_new = pc; k < 6; k++)
	{
	  for(n = 0; n < header1.npart[k]; n++)
	    {
	      fread(&Id[pc_new], sizeof(int), 1, fd);
	      pc_new++;
	    }
	}
      SKIP;


      if(ntot_withmasses > 0)
	SKIP;
      for(k = 0, pc_new = pc; k < 6; k++)
	{
	  for(n = 0; n < header1.npart[k]; n++)
	    {
	      P[pc_new].Type = k;

	      if(header1.mass[k] == 0)
		fread(&P[pc_new].Mass, sizeof(float), 1, fd);
	      else
		P[pc_new].Mass = header1.mass[k];
	      pc_new++;
	    }
	}
      if(ntot_withmasses > 0)
	SKIP;


      if(header1.npart[0] > 0)
	{
	  SKIP;
	  for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	    {
	      fread(&P[pc_sph].U, sizeof(float), 1, fd);
	      pc_sph++;
	    }
	  SKIP;

	  SKIP;
	  for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
	    {
	      fread(&P[pc_sph].Rho, sizeof(float), 1, fd);
	      pc_sph++;
	    }
	  SKIP;

	  SKIP;
		  for(n = 0, pc_sph = pc; n < header1.npart[0]; n++)
		{
		  fread(&P[pc_sph].Hsml, sizeof(float), 1, fd);
		  pc_sph++;
		}
		 SKIP;
	}

      fclose(fd);
    }


  Time = header1.time;
  Redshift = header1.time;
  return 0;
}

/* this routine allocates the memory for the
 * particle data.
 */
int allocate_memory(void)
{
  printf("allocating memory...\n");

  if(!(P = malloc(NumPart * sizeof(struct particle_data))))
    {
      fprintf(stderr, "failed to allocate memory.\n");
      exit(0);
    }

  P--;				/* start with offset 1 */


  if(!(Id = malloc(NumPart * sizeof(int))))
    {
      fprintf(stderr, "failed to allocate memory.\n");
      exit(0);
    }

  Id--;				/* start with offset 1 */

  printf("allocating memory...done\n");
  return 0;
}

/* This routine brings the particles back into
 * the order of their ID's.
 * NOTE: The routine only works if the ID's cover
 * the range from 1 to NumPart !
 * In other cases, one has to use more general
 * sorting routines.
 */
int reordering(void)
{
  int i;
  int idsource, idsave, dest;
  struct particle_data psave, psource;


  printf("reordering....\n");

  for(i = 1; i <= NumPart; i++)
    {
      if(Id[i] != i)
	{
	  psource = P[i];
	  idsource = Id[i];
	  dest = Id[i];

	  do
	    {
	      psave = P[dest];
	      idsave = Id[dest];

	      P[dest] = psource;
	      Id[dest] = idsource;

	      if(dest == i)
		break;

	      psource = psave;
	      idsource = idsave;

	      dest = idsource;
	    }
	  while(1);
	}
    }

  printf("done.\n");

  Id++;
  free(Id);

  printf("space for particle ID freed\n");
  return 0;
}
