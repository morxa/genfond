(define (problem problem-full-5-1)
    (:domain grid-visit-all)
    (:objects loc_x0_y0 loc_x0_y2 loc_x0_y4 loc_x1_y0 loc_x1_y3 loc_x1_y4 loc_x2_y2 loc_x2_y4 loc_x3_y0 loc_x3_y1 loc_x3_y2 loc_x3_y3 - place)
    (:init (at-robot loc_x0_y2) (connected loc_x0_y0 loc_x1_y0) (connected loc_x0_y4 loc_x1_y4) (connected loc_x1_y0 loc_x0_y0) (connected loc_x1_y3 loc_x1_y4) (connected loc_x1_y4 loc_x0_y4) (connected loc_x1_y4 loc_x1_y3) (connected loc_x1_y4 loc_x2_y4) (connected loc_x2_y2 loc_x3_y2) (connected loc_x2_y4 loc_x1_y4) (connected loc_x3_y0 loc_x3_y1) (connected loc_x3_y1 loc_x3_y0) (connected loc_x3_y1 loc_x3_y2) (connected loc_x3_y2 loc_x2_y2) (connected loc_x3_y2 loc_x3_y1) (connected loc_x3_y2 loc_x3_y3) (connected loc_x3_y3 loc_x3_y2) (visited loc_x0_y2))
    (:goal (and (visited loc_x0_y0) (visited loc_x0_y2) (visited loc_x0_y4) (visited loc_x1_y0) (visited loc_x1_y3) (visited loc_x1_y4) (visited loc_x2_y2) (visited loc_x2_y4) (visited loc_x3_y0) (visited loc_x3_y1) (visited loc_x3_y2) (visited loc_x3_y3)))
)