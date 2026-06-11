#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

struct point {
    int i;
    int j;    
};

void saveInitializedState() {
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        h_ACTIVITY_INIT[l] = h_ACTIVITY[l];

        h_Q_INIT[l] = h_Q[l];
        h_Q_INIT[NMAX + l] = h_Q[NMAX + l];

        h_U_INIT[l] = h_U[l];
        h_U_INIT[NMAX + l] = h_U[NMAX + l];
        h_U_INIT[2 * NMAX + l] = h_U[2 * NMAX + l];

        #pragma unroll
        for(int m = 0; m < LATTICE_VELOCITY_NUMBER; m++) {
            h_F_INIT[m * NMAX + l] = h_F[m * NMAX + l];
        }
    }
}

void resetState() {
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        h_ACTIVITY[l] = h_ACTIVITY_INIT[l];

        h_Q[l] = h_Q_INIT[l];
        h_Q[NMAX + l] = h_Q_INIT[NMAX + l];

        h_U[l] = h_U_INIT[l];
        h_U[NMAX + l] = h_U_INIT[NMAX + l];
        h_U[2 * NMAX + l] = h_U_INIT[2 * NMAX + l];

        #pragma unroll
        for(int m = 0; m < LATTICE_VELOCITY_NUMBER; m++) {
            h_F[m * NMAX + l] = h_F_INIT[m * NMAX + l];
        }
    }
}

//Write the density and velocity field to a file
void writeVelocity(int t) {
    char buf[120];
    sprintf(buf, "./%svelocity_%d.dat", OUTPUT_FILE_PREFIX, t);
    FILE *file;
    file = fopen(buf, "w");

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        fprintf(file, "%d %d %f %f %f\n", (l % I), ((l % (I * J)) / I), h_U[l], h_U[NMAX + l], h_U[2 * NMAX + l]);
    }
    
    fclose(file);
}

//Write the degree of order and director angle field to a file
void writeOrientation(int t) {
    char buf[120];
    sprintf(buf, "./%sorientation_%d.dat", OUTPUT_FILE_PREFIX, t);
    FILE *file;
    file = fopen(buf, "w");

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        float degree_of_order = sqrt(4 * (h_Q[l] * h_Q[l] + h_Q[NMAX + l] * h_Q[NMAX + l]));
        float angle = 0.5 * atan2(h_Q[NMAX + l], h_Q[l]);
        fprintf(file, "%d %d %f %f\n", (l % I), ((l % (I * J)) / I), degree_of_order, angle);
    }
    
    fclose(file);
}

void writeFreeEnergy(int t) {
    char buf[120];
    sprintf(buf, "./%sfree_energy_%d.dat", OUTPUT_FILE_PREFIX, t);
    FILE *file;
    file = fopen(buf, "w");

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        if (h_LMARK[l] == LMARK_BULK) {
            float trQ2 = 2 * (h_Q[l] * h_Q[l] + h_Q[NMAX + l] * h_Q[NMAX + l]);
            float trQ3 = 0;  // Only true for 2D nematics; nonzero for 3D

            // Remember that Qxx = -Qyy and Qxy = Qyx since Q is symmetric and traceless
            float dQxxdx = (h_Q[l + 1] - h_Q[l - 1]) / 2.0f;
            float dQxydx = (h_Q[NMAX + l + 1] - h_Q[NMAX + l - 1]) / 2.0f;
            float dQxxdy = (h_Q[l + I] - h_Q[l - I]) / 2.0f;
            float dQxydy = (h_Q[NMAX + l + I] - h_Q[NMAX + l - I]) / 2.0f;
            float grad_sq_Q = 2 * (dQxxdx * dQxxdx + dQxxdy * dQxxdy) + dQxydx * dQxydx + dQxydy * dQxydy;
            float bulk_free_energy = A / 2.0 * trQ2 + B / 3.0 * trQ3 + C / 4.0 * trQ2 * trQ2;
            float elastic_free_energy = L / 2.0 * grad_sq_Q;
            float free_energy = bulk_free_energy + elastic_free_energy;
            //float free_energy = A / 2.0 * trQ2 + B / 3.0 * trQ3 + C / 4.0 * trQ2 * trQ2 + L / 2.0 * grad_sq_Q;
            //fprintf(file, "%d %d %f\n", (l % I), ((l % (I * J)) / I), free_energy);
            fprintf(file, "%d %d %f %f %f\n", (l % I), ((l % (I * J)) / I), free_energy, bulk_free_energy, elastic_free_energy);
        } else {
            fprintf(file, "%d %d NaN NaN NaN\n", (l % I), ((l % (I * J)) / I));
        }
    }

    fclose(file);
}

void writeStress(int t) {
    char buf[120];
    sprintf(buf, "./%sstress_%d.dat", OUTPUT_FILE_PREFIX, t);
    FILE *file;
    file = fopen(buf, "w");

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        if (h_LMARK[l] == LMARK_BULK) {
            float Sxx = h_SIGMA[l];
            float Sxy = h_SIGMA[NMAX + l];
            float Syx = h_SIGMA[2 * NMAX + l];
            float Syy = h_SIGMA[3 * NMAX + l];
            float Sxy_sym = 0.5 * (Sxy + Syx);
            float isotropic_stress = -0.5 * (Sxx + Syy);
            float von_mises_stress = sqrt(Sxx * Sxx - Sxx * Syy + Syy * Syy + 3 * Sxy_sym * Sxy_sym);
            float principal_stress_diff = sqrt((Sxx - Syy) * (Sxx - Syy) + 4 * Sxy_sym * Sxy_sym);
            fprintf(file, "%d %d %f %f %f\n", (l % I), ((l % (I * J)) / I), isotropic_stress, von_mises_stress, principal_stress_diff);
        } else {
            fprintf(file, "%d %d NaN NaN NaN\n", (l % I), ((l % (I * J)) / I));
        }
    }

    fclose(file);
}

//Write goal_frac_i and goal_frac_j to a file
void writeGoal(int t) {
    char buf[120];
    sprintf(buf, "./%sgoal.param", INPUT_FILE_PREFIX);
    FILE *file;
    file = fopen(buf, "r");
    float goal_frac_i, goal_frac_j;

    if (file == NULL) {
        printf("Goal param file %s does not exist!\n", buf);
        exit(-1);
    }

    fscanf(file, "%f %f\n", &goal_frac_i, &goal_frac_j);
    fclose(file);

    sprintf(buf, "./%sgoal_%d.dat", OUTPUT_FILE_PREFIX, t);
    file = fopen(buf, "w");
    fprintf(file, "%f %f\n", goal_frac_i, goal_frac_j);
    fclose(file);
}

void writeObservations() {
    char buf[120];
    sprintf(buf, "./%svelocity.dat", OUTPUT_FILE_PREFIX);
    FILE *file;
    file = fopen(buf, "w");

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        fprintf(file, "%d %d %f %f %f\n", (l % I), ((l % (I * J)) / I), h_U[l], h_U[NMAX + l], h_U[2 * NMAX + l]);
    }

    fclose(file);

    sprintf(buf, "./%sorientation.dat", OUTPUT_FILE_PREFIX);
    file = fopen(buf, "w");

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        fprintf(file, "%d %d %f %f\n", (l % I), ((l % (I * J)) / I), h_Q[l], h_Q[NMAX + l]);
    }

    fclose(file);

    sprintf(buf, "./%sdefects.dat", OUTPUT_FILE_PREFIX);
    file = fopen(buf, "w");

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        fprintf(file, "%d %d %f\n", (l % I), ((l % (I * J)) / I), h_DEFECT_MAP[l]);
    }

    fclose(file);
}

void writeReward(float reward) {
    char buf[120];
    sprintf(buf, "./%sreward.dat", OUTPUT_FILE_PREFIX);
    FILE *file;
    file = fopen(buf, "w");
    fprintf(file, "%f\n", reward);
    fclose(file);
}

void readActivity() {
    char buf[120];
    sprintf(buf, "./%sactivity.dat", INPUT_FILE_PREFIX);
    FILE *file;
    file = fopen(buf, "r");
    int i, j;

    if (file == NULL) {
        printf("Activity input file %s does not exist!\n", buf);
        exit(-1);
    }

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        fscanf(file, "%d %d %f\n", &i, &j, h_ACTIVITY + l);
        assert(i == (l % I));
        assert(j == ((l % (I * J)) / I));
    }

    fclose(file);
}

void writeActivity(int t) {
    char buf[120];
    sprintf(buf, "./%sactivity_%d.dat", OUTPUT_FILE_PREFIX, t);
    FILE *file;
    file = fopen(buf, "w");

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        fprintf(file, "%d %d %f\n", (l % I), ((l % (I * J)) / I), h_ACTIVITY[l]);
    }

    fclose(file);
}

struct point rotate_point(int l, int i_center, int j_center, float angle) {
    struct point point_rot;
    int i = (l % I);
    int j = ((l % (I * J)) / I);
    int rel_i = i - i_center;
    //int rel_j = j - j_center;
    int rel_j = j_center - j;
    float cosTheta = cos(angle);
    float sinTheta = sin(angle);
    point_rot.i = round(i_center + cosTheta * rel_i - sinTheta * rel_j);
    point_rot.j = round(j_center - (sinTheta * rel_i + cosTheta * rel_j));
    return point_rot;
}

bool isPointInActivityPattern(int l) {
    int i = l % I;
    int j = (l % (I * J)) / I;

    // pattern along horizontal channel
    int in_top_col = round(12 / 33. * J);
    int in_bot_col = round(9 / 33. * J);
    int in_left_col = round(2 / 33. * I);
    int in_right_col = round(12 / 33. * I);

    // pattern along vertical channel
    int out_top_col = round(19 / 33. * J);
    int out_bot_col = round(12 / 33. * J);
    int out_left_col = round(9 / 33. * I);
    int out_right_col = round(12 / 33. * I);

    if (i >= in_left_col && i <= in_right_col && j >= in_bot_col && j <= in_top_col) {
        return true;
    }

    if (i >= out_left_col && i <= out_right_col && j >= out_bot_col && j <= out_top_col) {
        return true;
    }

    return false;
}

float computeActivityPatternFraction() {
    int num_active_sites = 0;
    int bulk_sites = 0;

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        if (h_ACTIVITY[l] > 0) {
            num_active_sites += 1;
        }

        if (h_LMARK[l] == LMARK_BULK) {
            bulk_sites += 1;
        }
    }

    return float(num_active_sites) / bulk_sites;
}

void sendPipeSignal(bool is_done) {
    FILE *send_fp = fopen(SEND_PIPE_PATH, "w");

    if (send_fp != NULL) {
        fputc(is_done ? 49 : 48, send_fp);  // ASCII int 48 is char "0", 49 is "1"
        fclose(send_fp);
        fflush(stdout);
    } else {
        printf("Send pipe file does not exist. Continuing...\n");
    }
}

bool recvPipeSignal() {
    FILE *recv_fp = fopen(RECV_PIPE_PATH, "r");

    if (recv_fp != NULL) {
        int reset = fgetc(recv_fp); // must be int, not char, to handle EOF
        fclose(recv_fp);
        fflush(stdout);

        if (reset == 49) {  // ASCII int 49 is char "1"
            return true;
        } else {
            return false;
        }
    } else {
        printf("Recv pipe file does not exist. Continuing...\n");
    }

    return false;
}
