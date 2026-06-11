#include <stdio.h>

void createFreeGeometry() {
    printf("Creating free geometry...\n");
    fflush(stdout);

    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        int i = l % I;
        int j = (l % (I * J)) / I;

        if (i == 0) { // outlets
            h_LMARK[l] = LMARK_LEFT_OUTLET_PBC;
        } else if (i == I - 1) {
            h_LMARK[l] = LMARK_RIGHT_OUTLET_PBC;
        } else if (j == 0) {
            h_LMARK[l] = LMARK_BOT_OUTLET_PBC;
        } else if (j == J - 1) {
            h_LMARK[l] = LMARK_TOP_OUTLET_PBC;
        } else {
            h_LMARK[l] = LMARK_BULK;
        }

        h_ACTIVITY[l] = ALPHA;  //0.0;

        // Density and velocity fields
        h_U[l] = DENSITYINIT;
        h_U[NMAX + l] = 0.0;
        h_U[2 * NMAX + l] = 0.0;

        // Q tensor
        float degree_of_order = 1.;
        float angle = 0.5 * M_PI;

        h_Q[l] = degree_of_order / 2.0 * cos(2 * angle);  //Qxx component
        h_Q[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  //Qxy component
        h_QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);  //Qxx component
        h_QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  //Qxy component
    }

    printf("Done creating free geometry\n");
    fflush(stdout);
}

void createSingleChannelGeometry() {
    printf("Creating single channel geometry...\n");
    fflush(stdout);

    // Channel and obstacle structure
    int bot_row = round(9 / 21. * J);
    int top_row = round(12 / 21. * J);

    // Initialize host arrays
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        int i = l % I;
        int j = (l % (I * J)) / I;

        //Logical markers
        if (i == 0 && (j < top_row && j > bot_row)) { // outlets
            h_LMARK[l] = LMARK_LEFT_OUTLET_PBC;
        } else if (i == I - 1 && j < top_row && j > bot_row) {
            h_LMARK[l] = LMARK_RIGHT_OUTLET_PBC;
        } else if (j == bot_row) { // walls
            h_LMARK[l] = LMARK_BOT_WALL;
        } else if (j == top_row) {
            h_LMARK[l] = LMARK_TOP_WALL;
        } else if (j < bot_row || j > top_row) { // bulk
            h_LMARK[l] = LMARK_OBS_BULK;
        } else {
            h_LMARK[l] = LMARK_BULK;
        }

        //if (isPointInActivityPattern(l)) h_ACTIVITY[l] = ALPHA;
        //else h_ACTIVITY[l] = 0.0;
        h_ACTIVITY[l] = 0.0;

        // Density and velocity fields
        h_U[l] = DENSITYINIT;
        h_U[NMAX + l] = 0.0;
        h_U[2 * NMAX + l] = 0.0;

        // Q tensor
        float degree_of_order = 1.;
        float angle = 0.5 * M_PI;

        h_Q[l] = degree_of_order / 2.0 * cos(2 * angle);  //Qxx component
        h_Q[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  //Qxy component
        h_QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);  //Qxx component
        h_QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  //Qxy component
    }

    printf("Done creating single channel geometry\n");
    fflush(stdout);
}

void createTJunctionGeometry() {
    printf("Creating T junction geometry...\n");
    fflush(stdout);

    // Channel and obstacle structure
    int bot_row = round(9 / 21. * J);
    int top_row = round(12 / 21. * J);
    int left_col = round(18 / 21. * I);
    int right_col = I - 1;

    // Initialize host arrays
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        int i = l % I;
        int j = (l % (I * J)) / I;

        //Logical markers
        if (i == 0 && j < top_row && j > bot_row) { // outlets
            h_LMARK[l] = LMARK_LEFT_OUTLET_OBC;
        } else if (i > left_col && i < right_col && j == J - 1) {
            h_LMARK[l] = LMARK_TOP_OUTLET_OBC;
        } else if (i > left_col && i < right_col && j == 0) {
            h_LMARK[l] = LMARK_BOT_OUTLET_OBC;
        } else if (j == bot_row && i < left_col) { // walls
            h_LMARK[l] = LMARK_BOT_WALL;
        } else if (j == top_row && i < left_col) {
            h_LMARK[l] = LMARK_TOP_WALL;
        } else if (i == left_col && (j < bot_row || j > top_row)) {
            h_LMARK[l] = LMARK_LEFT_WALL;
        } else if (i == right_col) {
            h_LMARK[l] = LMARK_RIGHT_WALL;
        } else if (i == left_col && j == bot_row) { // corners
            h_LMARK[l] = LMARK_CORNER_BOT_LEFT_25;
        } else if (i == left_col && j == top_row) {
            h_LMARK[l] = LMARK_CORNER_TOP_LEFT_25;
        } else if (i < left_col && (j < bot_row || j > top_row)) { // bulk
            h_LMARK[l] = LMARK_OBS_BULK;
        } else {
            h_LMARK[l] = LMARK_BULK;
        }

        //if ((i < right_col && j > bot_row && j < top_row) || (i > left_col && i < right_col && j >= top_row)) {
        //    h_ACTIVITY[l] = ALPHA;
        //}
        //if (isPointInActivityPattern(l)) h_ACTIVITY[l] = ALPHA;
        //else h_ACTIVITY[l] = 0.0;
        h_ACTIVITY[l] = 0.0;

        // Density and velocity fields
        h_U[l] = DENSITYINIT;
        h_U[NMAX + l] = 0.0;
        h_U[2 * NMAX + l] = 0.0;

        // Q tensor
        float degree_of_order = 1.;
        float angle = 0.25 * M_PI;

        if (i < left_col && j >= bot_row && j <= top_row) {
            angle = 0.5 * M_PI;
        } else if (i >= left_col && i <= right_col && (j < bot_row || j > top_row)) {
            angle = 0.0;
        }

        h_Q[l] = degree_of_order / 2.0 * cos(2 * angle);  // Qxx component
        h_Q[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  // Qxy component
        h_QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);  // Qxx component
        h_QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  // Qxy component
    }

    printf("Done creating T junction geometry\n");
    fflush(stdout);
}

void createCrossJunctionGeometry() {
    printf("Creating cross junction geometry...\n");
    fflush(stdout);

    // Channel and obstacle structure
    int bot_row = round(9 / 21. * J);
    int top_row = round(12 / 21. * J);
    int left_col = round(9 / 21. * I);
    int right_col = round(12 / 21. * I);

    // Initialize host arrays
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        int i = l % I;
        int j = (l % (I * J)) / I;

        //Logical markers
        if (i == 0 && j < top_row && j > bot_row) { // outlets
            h_LMARK[l] = LMARK_LEFT_OUTLET_PBC;
        } else if (i == I - 1 && j < top_row && j > bot_row) {
            h_LMARK[l] = LMARK_RIGHT_OUTLET_PBC;
        } else if (i > left_col && i < right_col && j == J - 1) {
            h_LMARK[l] = LMARK_TOP_OUTLET_PBC;
        } else if (i > left_col && i < right_col && j == 0) {
            h_LMARK[l] = LMARK_BOT_OUTLET_PBC;
        } else if (j == bot_row && (i < left_col || i > right_col)) { // walls
            h_LMARK[l] = LMARK_BOT_WALL;
        } else if (j == top_row && (i < left_col || i > right_col)) {
            h_LMARK[l] = LMARK_TOP_WALL;
        } else if (i == left_col && (j < bot_row || j > top_row)) {
            h_LMARK[l] = LMARK_LEFT_WALL;
        } else if (i == right_col && (j < bot_row || j > top_row)) {
            h_LMARK[l] = LMARK_RIGHT_WALL;
        } else if (i == left_col && j == bot_row) { // corners
            h_LMARK[l] = LMARK_CORNER_BOT_LEFT_25;
        } else if (i == left_col && j == top_row) {
            h_LMARK[l] = LMARK_CORNER_TOP_LEFT_25;
        } else if (i == right_col && j == bot_row) {
            h_LMARK[l] = LMARK_CORNER_BOT_RIGHT_25;
        } else if (i == right_col && j == top_row) {
            h_LMARK[l] = LMARK_CORNER_TOP_RIGHT_25;
        } else if ((i < left_col && j < bot_row) ||
                   (i < left_col && j > top_row) ||
                   (i > right_col && j < bot_row) ||
                   (i > right_col && j > top_row)) { // bulk
            h_LMARK[l] = LMARK_OBS_BULK;
        } else {
            h_LMARK[l] = LMARK_BULK;
        }

        //if (isPointInActivityPattern(l)) h_ACTIVITY[l] = ALPHA;
        //else h_ACTIVITY[l] = 0.0;
        h_ACTIVITY[l] = 0.0;

        // Density and velocity fields
        h_U[l] = DENSITYINIT;
        h_U[NMAX + l] = 0.0;
        h_U[2 * NMAX + l] = 0.0;

        // Q tensor
        float degree_of_order = 1.;
        float angle = 0.25 * M_PI;

        if ((i < left_col || i > right_col) && j >= bot_row && j <= top_row) {
            angle = 0.5 * M_PI;
        } else if (i >= left_col && i <= right_col && (j < bot_row || j > top_row)) {
            angle = 0.0;
        }

        h_Q[l] = degree_of_order / 2.0 * cos(2 * angle);  // Qxx component
        h_Q[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  // Qxy component
        h_QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);  // Qxx component
        h_QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  // Qxy component
    }

    printf("Done creating cross junction geometry\n");
    fflush(stdout);
}

void createCrossMazeCWGeometry() {
    printf("Creating cross maze clockwise geometry...\n");
    fflush(stdout);

    // Channel and obstacle structure
    int bot_row_south = round(9 / 33. * J);
    int top_row_south = round(12 / 33. * J);
    int bot_row_north = round(21 / 33. * J);
    int top_row_north = round(24 / 33. * J);
    int left_col_west = round(9 / 33. * I);
    int right_col_west = round(12 / 33. * I);
    int left_col_east = round(21 / 33. * I);
    int right_col_east = round(24 / 33. * I);

    // Initialize host arrays
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        int i = l % I;
        int j = (l % (I * J)) / I;

        //Logical markers
        if (i == 0 && ((j < top_row_south && j > bot_row_south) || (j < top_row_north && j > bot_row_north))) { // outlets
            h_LMARK[l] = LMARK_LEFT_OUTLET_PBC;
        } else if (i == I - 1 && ((j < top_row_south && j > bot_row_south) || (j < top_row_north && j > bot_row_north))) {
            h_LMARK[l] = LMARK_RIGHT_OUTLET_PBC;
        } else if (((i > left_col_west && i < right_col_west) || (i > left_col_east && i < right_col_east)) && j == J - 1) {
            h_LMARK[l] = LMARK_TOP_OUTLET_PBC;
        } else if (((i > left_col_west && i < right_col_west) || (i > left_col_east && i < right_col_east)) && j == 0) {
            h_LMARK[l] = LMARK_BOT_OUTLET_PBC;
        } else if ((j == bot_row_south || j == bot_row_north) &&
                   (i < left_col_west || (i > right_col_west && i < left_col_east) || i > right_col_east)) { // walls
            h_LMARK[l] = LMARK_BOT_WALL;
        } else if ((j == top_row_south || j == top_row_north) &&
                   (i < left_col_west || (i > right_col_west && i < left_col_east) || i > right_col_east)) {
            h_LMARK[l] = LMARK_TOP_WALL;
        } else if ((i == left_col_west || i == left_col_east) &&
                   (j < bot_row_south || (j > top_row_south && j < bot_row_north) || j > top_row_north)) {
            h_LMARK[l] = LMARK_LEFT_WALL;
        } else if ((i == right_col_west || i == right_col_east) &&
                   (j < bot_row_south || (j > top_row_south && j < bot_row_north) || j > top_row_north)) {
            h_LMARK[l] = LMARK_RIGHT_WALL;
        } else if ((i == left_col_west && j == bot_row_south) ||
                   (i == left_col_west && j == bot_row_north) ||
                   (i == left_col_east && j == bot_row_north)) { // corners
            h_LMARK[l] = LMARK_CORNER_BOT_LEFT_25;
        } else if (i == left_col_east && j == bot_row_south) {
            h_LMARK[l] = LMARK_CORNER_BOT_LEFT_75;
        } else if ((i == right_col_west && j == bot_row_south) ||
                   (i == right_col_west && j == bot_row_north) ||
                   (i == right_col_east && j == bot_row_north)) {
            h_LMARK[l] = LMARK_CORNER_BOT_RIGHT_25;
        } else if (i == right_col_east && j == bot_row_south) {
            h_LMARK[l] = LMARK_CORNER_BOT_RIGHT_75;
        } else if ((i == left_col_west && j == top_row_south) ||
                   (i == left_col_west && j == top_row_north) ||
                   (i == left_col_east && j == top_row_north)) {
            h_LMARK[l] = LMARK_CORNER_TOP_LEFT_25;
        } else if (i == left_col_east && j == top_row_south) {
            h_LMARK[l] = LMARK_CORNER_TOP_LEFT_75;
        } else if ((i == right_col_west && j == top_row_south) ||
                   (i == right_col_west && j == top_row_north) ||
                   (i == right_col_east && j == top_row_north)) {
            h_LMARK[l] = LMARK_CORNER_TOP_RIGHT_25;
        } else if (i == right_col_east && j == top_row_south) {
            h_LMARK[l] = LMARK_CORNER_TOP_RIGHT_75;
        } else if ((i < left_col_west && j < bot_row_south) ||
                   (i > right_col_east && j < bot_row_south) ||
                   (i < left_col_west && j > top_row_north) ||
                   (i > right_col_east && j > top_row_north) ||
                   (i < left_col_west && j > top_row_south && j < bot_row_north) ||
                   (i > right_col_east && j > top_row_south && j < bot_row_north) ||
                   (i > right_col_west && i < left_col_east && j < bot_row_south) ||
                   (i > right_col_west && i < left_col_east && j > top_row_north) ||
                   (i > right_col_west && i < left_col_east && j > top_row_south && j < bot_row_north)) { // bulk
            h_LMARK[l] = LMARK_OBS_BULK;
        } else {
            h_LMARK[l] = LMARK_BULK;
        }

        //if (isPointInActivityPattern(l)) h_ACTIVITY[l] = ALPHA;
        //else h_ACTIVITY[l] = 0.0;
        h_ACTIVITY[l] = 0.0;

        // Density and velocity fields
        h_U[l] = DENSITYINIT;
        h_U[NMAX + l] = 0.0;
        h_U[2 * NMAX + l] = 0.0;

        // Q tensor
        float degree_of_order = 1.;
        float angle = 0.25 * M_PI;

        if (i > left_col_east && i < right_col_east && j > bot_row_south && j < top_row_south) {
            angle = 0.75 * M_PI;
        } else if ((i < left_col_west || (i > right_col_west && i < left_col_east) || i > right_col_east) &&
            ((j > bot_row_south && j < top_row_south) || (j > bot_row_north && j < top_row_north))) {
            angle = 0.5 * M_PI;
        } else if (((i > left_col_west && i < right_col_west) || (i > left_col_east && i < right_col_east)) &&
                   (j < bot_row_south || (j > top_row_south && j < bot_row_north) || j > top_row_north)) {
            angle = 0.0;
        }

        h_Q[l] = degree_of_order / 2.0 * cos(2 * angle);  // Qxx component
        h_Q[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  // Qxy component
        h_QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);  // Qxx component
        h_QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  // Qxy component

        // NOTE:
        // Q = s * ([[ cos^2(theta) - 1/2,      cos(theta) * sin(theta) ],
        //           [ cos(theta) * sin(theta), sin^2(theta) - 1/2      ]])
        //   = (s/2) * ([[ cos(2 * theta),  sin(2 * theta)]],
        //               [ sin(2 * theta), -cos(2 * theta)]])      // using trig identities
        //
        // We only store two values for Q (Q11 and Q12) since Q22 = -Q11 and Q12 = Q21
    }

    printf("Done creating cross maze clockwise geometry\n");
    fflush(stdout);
}

void createCrossMazeCCWGeometry() {
    printf("Creating cross maze counterclockwise geometry...\n");
    fflush(stdout);

    // Channel and obstacle structure
    int bot_row_south = round(9 / 33. * J);
    int top_row_south = round(12 / 33. * J);
    int bot_row_north = round(21 / 33. * J);
    int top_row_north = round(24 / 33. * J);
    int left_col_west = round(9 / 33. * I);
    int right_col_west = round(12 / 33. * I);
    int left_col_east = round(21 / 33. * I);
    int right_col_east = round(24 / 33. * I);

    // Initialize host arrays
    #pragma unroll
    for (int l = 0; l < NMAX; l++) {
        int i = l % I;
        int j = (l % (I * J)) / I;

        //Logical markers
        if (i == 0 && ((j < top_row_south && j > bot_row_south) || (j < top_row_north && j > bot_row_north))) { // outlets
            h_LMARK[l] = LMARK_LEFT_OUTLET_PBC;
        } else if (i == I - 1 && ((j < top_row_south && j > bot_row_south) || (j < top_row_north && j > bot_row_north))) {
            h_LMARK[l] = LMARK_RIGHT_OUTLET_PBC;
        } else if (((i > left_col_west && i < right_col_west) || (i > left_col_east && i < right_col_east)) && j == J - 1) {
            h_LMARK[l] = LMARK_TOP_OUTLET_PBC;
        } else if (((i > left_col_west && i < right_col_west) || (i > left_col_east && i < right_col_east)) && j == 0) {
            h_LMARK[l] = LMARK_BOT_OUTLET_PBC;
        } else if ((j == bot_row_south || j == bot_row_north) &&
                   (i < left_col_west || (i > right_col_west && i < left_col_east) || i > right_col_east)) { // walls
            h_LMARK[l] = LMARK_BOT_WALL;
        } else if ((j == top_row_south || j == top_row_north) &&
                   (i < left_col_west || (i > right_col_west && i < left_col_east) || i > right_col_east)) {
            h_LMARK[l] = LMARK_TOP_WALL;
        } else if ((i == left_col_west || i == left_col_east) &&
                   (j < bot_row_south || (j > top_row_south && j < bot_row_north) || j > top_row_north)) {
            h_LMARK[l] = LMARK_LEFT_WALL;
        } else if ((i == right_col_west || i == right_col_east) &&
                   (j < bot_row_south || (j > top_row_south && j < bot_row_north) || j > top_row_north)) {
            h_LMARK[l] = LMARK_RIGHT_WALL;
        } else if ((i == left_col_west && j == bot_row_south) ||
                   (i == left_col_west && j == bot_row_north) ||
                   (i == left_col_east && j == bot_row_south)) { // corners
            h_LMARK[l] = LMARK_CORNER_BOT_LEFT_25;
        } else if (i == left_col_east && j == bot_row_north) {
            h_LMARK[l] = LMARK_CORNER_BOT_LEFT_75;
        } else if ((i == right_col_west && j == bot_row_south) ||
                   (i == right_col_west && j == bot_row_north) ||
                   (i == right_col_east && j == bot_row_south)) {
            h_LMARK[l] = LMARK_CORNER_BOT_RIGHT_25;
        } else if (i == right_col_east && j == bot_row_north) {
            h_LMARK[l] = LMARK_CORNER_BOT_RIGHT_75;
        } else if ((i == left_col_west && j == top_row_south) ||
                   (i == left_col_west && j == top_row_north) ||
                   (i == left_col_east && j == top_row_south)) {
            h_LMARK[l] = LMARK_CORNER_TOP_LEFT_25;
        } else if (i == left_col_east && j == top_row_north) {
            h_LMARK[l] = LMARK_CORNER_TOP_LEFT_75;
        } else if ((i == right_col_west && j == top_row_south) ||
                   (i == right_col_west && j == top_row_north) ||
                   (i == right_col_east && j == top_row_south)) {
            h_LMARK[l] = LMARK_CORNER_TOP_RIGHT_25;
        } else if (i == right_col_east && j == top_row_north) {
            h_LMARK[l] = LMARK_CORNER_TOP_RIGHT_75;
        } else if ((i < left_col_west && j < bot_row_south) ||
                   (i > right_col_east && j < bot_row_south) ||
                   (i < left_col_west && j > top_row_north) ||
                   (i > right_col_east && j > top_row_north) ||
                   (i < left_col_west && j > top_row_south && j < bot_row_north) ||
                   (i > right_col_east && j > top_row_south && j < bot_row_north) ||
                   (i > right_col_west && i < left_col_east && j < bot_row_south) ||
                   (i > right_col_west && i < left_col_east && j > top_row_north) ||
                   (i > right_col_west && i < left_col_east && j > top_row_south && j < bot_row_north)) { // bulk
            h_LMARK[l] = LMARK_OBS_BULK;
        } else {
            h_LMARK[l] = LMARK_BULK;
        }

        //if (isPointInActivityPattern(l)) h_ACTIVITY[l] = ALPHA;
        //else h_ACTIVITY[l] = 0.0;
        h_ACTIVITY[l] = 0.0;

        // Density and velocity fields
        h_U[l] = DENSITYINIT;
        h_U[NMAX + l] = 0.0;
        h_U[2 * NMAX + l] = 0.0;

        // Q tensor
        float degree_of_order = 1.;
        float angle = 0.25 * M_PI;

        if (i > left_col_east && i < right_col_east && j > bot_row_north && j < top_row_north) {
            angle = 0.75 * M_PI;
        } else if ((i < left_col_west || (i > right_col_west && i < left_col_east) || i > right_col_east) &&
            ((j > bot_row_south && j < top_row_south) || (j > bot_row_north && j < top_row_north))) {
            angle = 0.5 * M_PI;
        } else if (((i > left_col_west && i < right_col_west) || (i > left_col_east && i < right_col_east)) &&
                   (j < bot_row_south || (j > top_row_south && j < bot_row_north) || j > top_row_north)) {
            angle = 0.0;
        }

        h_Q[l] = degree_of_order / 2.0 * cos(2 * angle);  // Qxx component
        h_Q[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  // Qxy component
        h_QNEW[l] = degree_of_order / 2.0 * cos(2 * angle);  // Qxx component
        h_QNEW[NMAX + l] = degree_of_order / 2.0 * sin(2 * angle);  // Qxy component

        // NOTE:
        // Q = s * ([[ cos^2(theta) - 1/2,      cos(theta) * sin(theta) ],
        //           [ cos(theta) * sin(theta), sin^2(theta) - 1/2      ]])
        //   = (s/2) * ([[ cos(2 * theta),  sin(2 * theta)]],
        //               [ sin(2 * theta), -cos(2 * theta)]])      // using trig identities
        //
        // We only store two values for Q (Q11 and Q12) since Q22 = -Q11 and Q12 = Q21
    }

    printf("Done creating cross maze counterclockwise geometry\n");
    fflush(stdout);
}

void createGeometry() {
    if (GEOM_NAME == "FREE") {
        createFreeGeometry();
    } else if (GEOM_NAME == "SINGLE_CHANNEL") {
        createSingleChannelGeometry();
    } else if (GEOM_NAME == "T_JUNCTION") {
        createTJunctionGeometry();
    } else if (GEOM_NAME == "CROSS_JUNCTION") {
        createCrossJunctionGeometry();
    } else if (GEOM_NAME == "CROSS_MAZE_CW") {
        createCrossMazeCWGeometry();
    } else if (GEOM_NAME == "CROSS_MAZE_CCW") {
        createCrossMazeCCWGeometry();
    } else {
        printf("ERROR: Unsupported GEOM_NAME %s", GEOM_NAME);
        fflush(stdout);
    }
}