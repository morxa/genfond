(define (problem visitall-5-2-4)
    (:domain grid-visit-all)
    (:requirements :typing)
    (:objects loc_x0_y0 loc_x0_y1 loc_x1_y0 loc_x1_y1 loc_x2_y0 loc_x2_y1 loc_x3_y0 loc_x3_y1 loc_x4_y0 loc_x4_y1 - place)
    (:init (at-robot loc_x0_y0) (connected loc_x0_y0 loc_x0_y1) (connected loc_x0_y0 loc_x1_y0) (connected loc_x0_y1 loc_x0_y0) (connected loc_x0_y1 loc_x1_y1) (connected loc_x1_y0 loc_x0_y0) (connected loc_x1_y0 loc_x1_y1) (connected loc_x1_y0 loc_x2_y0) (connected loc_x1_y1 loc_x0_y1) (connected loc_x1_y1 loc_x1_y0) (connected loc_x1_y1 loc_x2_y1) (connected loc_x2_y0 loc_x1_y0) (connected loc_x2_y0 loc_x2_y1) (connected loc_x2_y0 loc_x3_y0) (connected loc_x2_y1 loc_x1_y1) (connected loc_x2_y1 loc_x2_y0) (connected loc_x2_y1 loc_x3_y1) (connected loc_x3_y0 loc_x2_y0) (connected loc_x3_y0 loc_x3_y1) (connected loc_x3_y0 loc_x4_y0) (connected loc_x3_y1 loc_x2_y1) (connected loc_x3_y1 loc_x3_y0) (connected loc_x3_y1 loc_x4_y1) (connected loc_x4_y0 loc_x3_y0) (connected loc_x4_y0 loc_x4_y1) (connected loc_x4_y1 loc_x3_y1) (connected loc_x4_y1 loc_x4_y0) (visited loc_x0_y0))
    (:goal (and (visited loc_x0_y0) (visited loc_x0_y1) (visited loc_x1_y0) (visited loc_x1_y1) (visited loc_x2_y0) (visited loc_x2_y1) (visited loc_x3_y0) (visited loc_x3_y1) (visited loc_x4_y0) (visited loc_x4_y1)))
)