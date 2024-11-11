(define (problem visitall-1-4-1)
    (:domain grid-visit-all)
    (:requirements :typing)
    (:objects loc_x0_y0 loc_x0_y1 loc_x0_y2 loc_x0_y3 - place)
    (:init (at-robot loc_x0_y3) (connected loc_x0_y0 loc_x0_y1) (connected loc_x0_y1 loc_x0_y0) (connected loc_x0_y1 loc_x0_y2) (connected loc_x0_y2 loc_x0_y1) (connected loc_x0_y2 loc_x0_y3) (connected loc_x0_y3 loc_x0_y2) (visited loc_x0_y3))
    (:goal (and (visited loc_x0_y0) (visited loc_x0_y1) (visited loc_x0_y2) (visited loc_x0_y3)))
)