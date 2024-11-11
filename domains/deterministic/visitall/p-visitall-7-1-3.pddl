(define (problem visitall-7-1-3)
    (:domain grid-visit-all)
    (:requirements :typing)
    (:objects loc_x0_y0 loc_x1_y0 loc_x2_y0 loc_x3_y0 loc_x4_y0 loc_x5_y0 loc_x6_y0 - place)
    (:init (at-robot loc_x1_y0) (connected loc_x0_y0 loc_x1_y0) (connected loc_x1_y0 loc_x0_y0) (connected loc_x1_y0 loc_x2_y0) (connected loc_x2_y0 loc_x1_y0) (connected loc_x2_y0 loc_x3_y0) (connected loc_x3_y0 loc_x2_y0) (connected loc_x3_y0 loc_x4_y0) (connected loc_x4_y0 loc_x3_y0) (connected loc_x4_y0 loc_x5_y0) (connected loc_x5_y0 loc_x4_y0) (connected loc_x5_y0 loc_x6_y0) (connected loc_x6_y0 loc_x5_y0) (visited loc_x1_y0))
    (:goal (and (visited loc_x0_y0) (visited loc_x1_y0) (visited loc_x2_y0) (visited loc_x3_y0) (visited loc_x4_y0) (visited loc_x5_y0) (visited loc_x6_y0)))
)