/*
 ** This program determines the planet and disk mass for a simulation output
 ** using the algorithm described in Canup, Ward and Cameron 2001.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include "tipsy.h"
#include "disk-finder.h"

// Initialize data structzre
DFDATA *dfInitData(char *pchInFile)
{
		DFDATA *data;

		// Initialize memory
		data = malloc(sizeof(DFDATA));
		assert(data != NULL);

        data->comp = malloc(3*sizeof(double));
		assert(data->comp != NULL);

		data->vcomp = malloc(3*sizeof(double));
		assert(data->vcomp != NULL);
		
		data->Mp = 0.0;
		data->Md = 0.0;
		data->Mesc = 0.0;

		TipsyInitialize(&data->in,0,"stdin");
		
		data->N = iTipsyNumParticles(data->in);

		data->iGroup = malloc(data->N*sizeof(int));
		assert(data->iGroup != NULL);

		return(data);
}

// Read all particle from the tipsy file and initialize iGroup and N
void dfReadData(DFDATA *data)
{
		int i;
		assert(data != NULL);
		TipsyReadAll(data->in);

		// No particle belongs to a group
		for (i=0;i<data->N;i++)
		{
			data->iGroup[i] = 1;
		}
}

void dfFinalizeData(DFDATA *data)
{
		TipsyFinish(data->in);

		free(data->comp);
		free(data->vcomp);
		free(data);
}

void dfPrintMark(DFDATA *data)
{
		int i;
		// Print header
		printf("%i %i 0\n", data->N, data->N);	

		// Print iGroup for all particles
		for (i = 0; i < data->N; i++)
        {
			printf("%i\n", data->iGroup[i]);
		}
}

void Planet_Calc_COM(DFDATA *data, double *pMass)
{
		// Determine the COM for all particles that belong to the planet
		double ipMass;
		int i,j;

		// Make sure that the particles have been loaded
		assert(data != NULL);
		assert(data->in != NULL);
		assert(data->iGroup != NULL);
				
		// Calculate COM
		*pMass = 0.0;

		for (j = 0; j < 3; j++)
		{
			data->comp[j] = 0.0;
			data->vcomp[j] = 0.0;
		}
        
		for (i = 0; i < data->N; i++)
        {
			if (data->iGroup[i] != 1)
			{
//				fprintf(stderr, "%i\n", data->iGroup[i]);
				continue;
			}

			assert(data->iGroup[i] == 1);

			for (j = 0; j < 3; j++)
			{
				data->comp[j] += data->in->gp[i].mass*data->in->gp[i].pos[j];
				data->vcomp[j] += data->in->gp[i].mass*data->in->gp[i].vel[j];
			}
//			fprintf(stderr,"i: %i mass: %g\n",i,in->gp[i].mass);
			*pMass += data->in->gp[i].mass;
		}
		
		ipMass = *pMass;
//		fprintf(stderr,"ipMass: %g %g\n",ipMass,*pMass);
		for (j = 0; j < 3; j++)
		{
			data->comp[j] *= 1.0/ipMass;
			data->vcomp[j] *= 1.0/ipMass;
		}
        printf("comp, vcomp: [%f %f %f], [%f %f %f]\n",data->comp[0],data->comp[1],data->comp[2],data->vcomp[0],data->vcomp[1],data->vcomp[2]);
}

void CalcInitialMp(DFDATA *data, double rhop)
{
		double Mp;
		int i;

		Mp = 0.0;

		for (i=0;i<data->N;i++)
		{
			if (data->in->gp[i].rho > rhop)
			{
				Mp += data->in->gp[i].mass;
				data->iGroup[i] = 1;    //这里的iGroup[i] = 1 表示属于planet
			} else {
				// The particles does not belong to an group
				data->iGroup[i] = 0;
			}
		}

		data->Mp = Mp;
}

/*
** Check, if a particle is bound, so if
** E = Ekin - Epot < 0
*/
int bParticleBound(struct gas_particle *gp, double Mp, double *rp, double *vp)
{
}

/*
** Calculate a particles total energy.
*/
double dParticleEtot(struct gas_particle *gp, double Mp, double *rp, double *vp)
{
		double r,v2;
		int j;
		
		
		v2 = 0.0;
		for (j=0;j<3;j++)
		{
			v2 += (gp->vel[j]-vp[j])*(gp->vel[j]-vp[j]);
		}

		r = 0.0;
		for (j=0;j<3;j++)
		{
			r += (gp->pos[j]-rp[j])*(gp->pos[j]-rp[j]);
		}
		r = sqrt(r);
		
		// In our units G=1
		return (0.5*v2 - Mp/r);
}

/*
** Determine, if a particle is bound and if it is, calculate its
** orbital elements(轨道元素) from:
**
** -G*M/(2*a) = Etot (Etot: specific total energy)  动量定理
**
** j = sqrt(G*M*a*(1.0-e*e)) (j: specific total angular momentum(角动量))
*/
void CalcOrbitalElements(struct gas_particle *gp, double E, double j, double Mp, double *a, double *e)
{
		assert(E < 0.0);
	
		// Assume G=1	
		*a = -Mp/(2.0*E);       //majora
		*e = sqrt(1.0-j*j/(Mp*(*a)));
}

//void CalcRadius(DFDATA *data)    //计算半径
//{
//		data->R = cbrt(3.0*M/(4.0*M_PI*data->rhomean));

/*
** Do one iteration of the disk finder algorithm described in Canuo 2001.
*/
void DoDiskAnalysis(DFDATA *data)
{
		double Mp,Md,Mesc;
		double Epart,jpart, a, e;
		// A particles position and velocity relative to the planets COM
		double *r,*v;
		double R;
		int i,j;

		r = malloc(3*sizeof(double));
		assert(r != NULL);
		v = malloc(3*sizeof(double));
		assert(v != NULL);

        printf("-------------------------------new round------------------------------\n");

		// Estimate thr planets radius		
		data->R = cbrt(3.0*data->Mp/(4.0*M_PI*data->rhomean));
		
		fprintf(stderr,"Rp= %g\n", data->R);
        printf("Caculate Planet Radius R = %f\n",data->R);
		// Determine COM	
		Planet_Calc_COM(data, &Mp);

		Mp = 0.0;
		Md = 0.0;
		Mesc = 0.0;

		for (i=0;i<data->N;i++)
		{
			// r = Rp - Rcomp
			R = 0.0;
			
			for (j=0;j<3;j++)
			{
				r[j] = data->in->gp[i].pos[j]-data->comp[j];
				v[j] = data->in->gp[i].vel[j]-data->vcomp[j];
				R += r[j]*r[j];
			}
			R = sqrt(R);

            printf("\n1.weight mean for pos %d [ %f %f %f ] -> r [ %f %f %f ] -- [ %f %f %f]\n",i,data->in->gp[i].pos[0],data->in->gp[i].pos[1],data->in->gp[i].pos[2],r[0],r[1],r[2],data->comp[0],data->comp[1],data->comp[2]);
            printf("2.weight mean for vel %d [ %f %f %f ] -> v [ %f %f %f ] -- [ %f %f %f]\n",i,data->in->gp[i].vel[0],data->in->gp[i].vel[1],data->in->gp[i].vel[2],r[0],r[1],r[2],data->vcomp[0],data->vcomp[1],data->vcomp[2]);
            printf("3.planet R's %f, particle's R %f\n",data->R,R);

			if (R <= data->R)
			{
				// Planet particle
				data->iGroup[i] = 1;
				Mp += data->in->gp[i].mass;
                printf("4.++add particle's Mp %f -> planet %f\n",data->in->gp[i].mass,Mp);

			} else {
				// Determine a particles totala energy
				Epart = dParticleEtot(&data->in->gp[i], data->Mp, data->comp, data->vcomp);
//				fprintf(stderr,"Epart= %g\n", Epart);
				if (Epart < 0)
				{
					// Particle is bound, calculate orbital elements
					jpart = (r[1]*v[2]-r[2]*v[1])*(r[1]*v[2]-r[2]*v[1]);
					jpart += (r[2]*v[0]-r[0]*v[2])*(r[2]*v[0]-r[0]*v[2]);
					jpart += (r[0]*v[1]-r[1]*v[0])*(r[0]*v[1]-r[1]*v[0]);
					jpart =sqrt(jpart);

					CalcOrbitalElements(&data->in->gp[i], Epart, jpart, data->Mp,&a, &e);
					
					if (a*(1.0-e) > data->R)
					{
                        printf("5.--jpart %f, epart %f, a %f, e %f\n",jpart,Epart,a,e);
						// Periapsis > Rplanet, so its a disc particle
						data->iGroup[i]=2;
						Md += data->in->gp[i].mass;
					} else {
						// Particle belong to the planet
                        printf("6.+-jpart %f, epart %f, a %f, e %f\n",jpart,Epart,a,e);
						data->iGroup[i]=1;
						Mp += data->in->gp[i].mass;
					}
				} else {
					// Escaping particle
                    printf("7.!!particle belongs to escape.\n");
					assert(Epart >= 0.0);
					data->iGroup[i] = 3;
					Mesc += data->in->gp[i].mass;
				}
			}
		}

		// Set new values
		data->Mp = Mp;
		data->Md = Md;
		data->Mesc = Mesc;

		free(r);
		free(v);
}

void main(int argc, char **argv) {
		// Tipsy library
		// TCTX in;
		DFDATA *data;
		// Earth's mean density (5.6 g/cm3) in codeunits
		double rhoe = 15.20;
		// A particles specific orbital energy and angular momentum
		double E, j_part;
		double Mp_old;
		double mass;
        int i,j;
        FILE *fp;

		// Check command line arguments
		if (argc != 1) {
			fprintf(stderr,"Usage: disk-finder <input.std>\n");
			exit(1);
		}

		data = dfInitData("stdin");

		// Read all particles
		dfReadData(data);

        //dump into file
        fp = fopen("disk-finder.dump.txt","w");
        fprintf(fp,"pos[0] pos[1] pos[2] rho mass id vel[0] vel[1] vel[2]\n");
        for(i=0;i<data->N;i++)
        {
            fprintf(fp,"%f %f %f ",data->in->gp[i].pos[0],data->in->gp[i].pos[1],data->in->gp[i].pos[2]);
            fprintf(fp,"%f %f %d ",data->in->gp[i].rho,data->in->gp[i].mass,i);
            fprintf(fp,"%f %f %f\n",data->in->gp[i].vel[0],data->in->gp[i].vel[1],data->in->gp[i].vel[2]);
        }
        fclose(fp);

		
		CalcInitialMp(data, rhoe);   // 和python代码里不一样，python里取的是rho0
		//相当于python代码里的pgrp[np.where(rho > rho0)] = 2.

		Planet_Calc_COM(data, &mass); 

		// Debug information
# if 0
		Planet_Calc_COM(data, &mass);
		fprintf(stderr,"COM: ");
		for (j = 0; j < 3; j++)
		{
//			com[j] *= 1.0/mass;
//			vcom[j] *= 1.0/mass;
			fprintf(stderr,"%.6e ",data->comp[j]);
		}
		fprintf(stderr,"\n");
		fprintf(stderr,"VCOM: ");
		for (j = 0; j < 3; j++)
		{
			fprintf(stderr,"%.6e ",data->vcomp[j]);
		}
		fprintf(stderr," M=%.6e\n",mass);

		fprintf(stderr," M=%.6e\n",data->Mp);

		dfPrintMark(data);
#endif

		Mp_old = 0.0; // Mp0 = 0.
		data->rhomean = rhoe;

		while(fabs(Mp_old-data->Mp) > 1e-3)
		{
			Mp_old = data->Mp;	
			DoDiskAnalysis(data);

			fprintf(stderr,"Mp_old= %g Mp= %g Md= %g Mesc= %g\n",Mp_old,data->Mp,data->Md,data->Mesc);
		}
		
		//dfPrintMark(data);	
		dfFinalizeData(data);
}



