struct goal {
    float goal_frac_i;
    float goal_frac_j;
};

struct goal readGoal() {
    char buf[120];
    sprintf(buf, "./%sgoal.param", INPUT_FILE_PREFIX);
    FILE *file;
    file = fopen(buf, "r");
    float goal_frac_i, goal_frac_j;
    struct goal new_goal;

    if (file == NULL) {
        printf("Goal param file %s does not exist!\n", buf);
        exit(-1);
    }

    fscanf(file, "%f %f\n", &goal_frac_i, &goal_frac_j);
    assert(goal_frac_i >= 0.);
    assert(goal_frac_i <= 1.);
    assert(goal_frac_j >= 0.);
    assert(goal_frac_j <= 1.);
    fclose(file);

    new_goal.goal_frac_i = goal_frac_i;
    new_goal.goal_frac_j = goal_frac_j;

    return new_goal;
}

void addDefect() {
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        int i = l % I;
        int j = (l % (I * J)) / I;

        // defect location
        float defect_x = (GEOM_NAME == "T_JUNCTION") ? 209.5 : 9.5;    //I / 42.0 - 0.5; // 9.5;
        float defect_y = (GEOM_NAME == "CROSS_MAZE_CW") ? 450.5 : 209.5;  //209.5;  //0.5 * J - 0.5; //209.5;
        float defect_size_x = I / 3.5;
        float defect_size_y = J / 10.5;

        float dy_defect = (float)(j) - defect_y;

        if (dy_defect > 0.5 * J) {
            dy_defect -= J;
        }

        if (dy_defect < -0.5 * J) {
            dy_defect += J;
        }

        if (abs(dy_defect) <= 0.5 * defect_size_y) {
            float dx_defect = (float)(i) - defect_x;

            if (dx_defect > 0.5 * I) {
                dx_defect -= I;
            }

            if (dx_defect < -0.5 * I) {
                dx_defect += I;
            }

            // Set Q tensor
            if (abs(dx_defect) <= 0.5 * defect_size_x) {
                float angle = -dy_defect / defect_size_y * M_PI;
                float degree_of_order = 1.;  // degree of order (s) must be between -1/2 and 1
                h_Q[l] = degree_of_order / 2.0 * cos(2 * angle);  //Qxx component
                h_Q[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  //Qxy component
            }
        }
    }
}

float getDirectorAngle(int l) {
    float angle = 0.5 * atan2(h_Q[NMAX + l], h_Q[l]);

    // director angle must lie between 0 and pi
    if (angle < 0) {
        angle += M_PI;
    } else if (angle > M_PI) {
        angle -= M_PI;
    }

    return angle;
}

float getNematicOrder(int l) {
    return sqrt(4 * (h_Q[l] * h_Q[l] + h_Q[NMAX + l] * h_Q[NMAX + l]));
}

float computeAngleDiff(float startAngle, float endAngle) {
    float diff_candidates[] = {endAngle - startAngle, endAngle - startAngle + (float)M_PI, endAngle - startAngle - (float)M_PI};
    int min_abs_index = 0;
    float min_abs_value = abs(diff_candidates[0]);

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        if (abs(diff_candidates[i]) < min_abs_value) {
            min_abs_value = abs(diff_candidates[i]);
            min_abs_index = i;
        }
    }

    return diff_candidates[min_abs_index];
}

void appendDefectInfoToFile(int defect_idx) {
    char buf[120];
    sprintf(buf, "./%sdefect_obs.dat", OUTPUT_FILE_PREFIX);
    FILE *file;
    file = fopen(buf, "a");

    if (defect_idx == -1) {
        // no defect found
        fprintf(file, "%d %d %f %f %f %f\n", -1, -1, -1, -1, -1, -1);
    } else {
        float angle = 0.5 * atan2(h_Q[NMAX + defect_idx], h_Q[defect_idx]);

        // ensure director angle lies between 0 and pi
        if (angle < 0) {
            angle += M_PI;
        } else if (angle > M_PI) {
            angle -= M_PI;
        }

        fprintf(file, "%d %d %f %f %f %f\n", (defect_idx % I), ((defect_idx % (I * J)) / I), h_U[NMAX + defect_idx], h_U[2 * NMAX + defect_idx], cos(angle), sin(angle));
    }

    fclose(file);
}

void clearDefectInfoFile() {
    char buf[120];
    sprintf(buf, "./%sdefect_obs.dat", OUTPUT_FILE_PREFIX);
    FILE *file;
    file = fopen(buf, "w");  // clear file contents
    fclose(file);
}

void clearDefectMap() {
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        h_DEFECT_MAP[l] = 0.0;
    }
}

float findDefects() {
    clearDefectInfoFile();
    clearDefectMap();
    struct goal current_goal = readGoal();
    int defect_goal_i = round(current_goal.goal_frac_i * I);
    int defect_goal_j = round(current_goal.goal_frac_j * J);
    printf("Current goal set to i: %d, j: %d\n", defect_goal_i, defect_goal_j);

    bool isNoDefectFound = true;
    float closest_defect_idx = -1;
    float dist_from_goal = 1e+03;

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        if (l + I + 1 < NMAX && l - I - 1 >= 0) {  // check that neighborhood of l lies entirely inside the simulation domain
            float defect_order = getNematicOrder(l);

            if (defect_order <= DEFECT_ORDER_THRESH) {
                float angle = getDirectorAngle(l);

                int i1 = l + I - 1;
                int i2 = l + I;
                int i3 = l + I + 1;
                int i4 = l - 1;
                int i5 = l + 1;
                int i6 = l - I - 1;
                int i7 = l - I;
                int i8 = l - I + 1;

                if (defect_order < getNematicOrder(i1) &&
                    defect_order < getNematicOrder(i2) &&
                    defect_order < getNematicOrder(i3) &&
                    defect_order < getNematicOrder(i4) &&
                    defect_order < getNematicOrder(i5) &&
                    defect_order < getNematicOrder(i6) &&
                    defect_order < getNematicOrder(i7) &&
                    defect_order < getNematicOrder(i8)) {
                    // nematic order at center is less than all neighbors
                    int i = l % I;
                    int j = (l % (I * J)) / I;
                    float winding_number = 0;
                    winding_number += computeAngleDiff(getDirectorAngle(i5), getDirectorAngle(i3));
                    winding_number += computeAngleDiff(getDirectorAngle(i3), getDirectorAngle(i2));
                    winding_number += computeAngleDiff(getDirectorAngle(i2), getDirectorAngle(i1));
                    winding_number += computeAngleDiff(getDirectorAngle(i1), getDirectorAngle(i4));
                    winding_number += computeAngleDiff(getDirectorAngle(i4), getDirectorAngle(i6));
                    winding_number += computeAngleDiff(getDirectorAngle(i6), getDirectorAngle(i7));
                    winding_number += computeAngleDiff(getDirectorAngle(i7), getDirectorAngle(i8));
                    winding_number += computeAngleDiff(getDirectorAngle(i8), getDirectorAngle(i5));
                    winding_number = round(10 * winding_number / (2 * M_PI)) / 10.; // divide winding_number by 2 * pi and round to 1 decimal place

                    if (abs(winding_number - 0.5) < 0.01) {
                        printf("Found +1/2 defect at i: %d, j: %d\n", i, j);
                        isNoDefectFound = false;
                        h_DEFECT_MAP[l] = 1.0;
                        appendDefectInfoToFile(l);
                        float dist_defect_from_goal = sqrt((i - defect_goal_i) * (i - defect_goal_i) + (j - defect_goal_j) * (j - defect_goal_j));

                        if (dist_defect_from_goal < dist_from_goal) {
                            closest_defect_idx = l;
                            dist_from_goal = dist_defect_from_goal;
                            printf("Updated dist_from_goal: %.1f\n", dist_from_goal);
                        }
                    } else if (abs(winding_number + 0.5) < 0.01) {
                        printf("Found -1/2 defect at i: %d, j: %d\n", i, j);
                        h_DEFECT_MAP[l] = -1.0;
                    }
                }
            }
        }
    }

    if (isNoDefectFound) {
        appendDefectInfoToFile(-1);
    }
    return dist_from_goal;
}