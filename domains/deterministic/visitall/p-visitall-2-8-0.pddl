(define (problem visitall-2-8-0)
    (:domain grid-visit-all)
    (:requirements :typing)
    (:objects loc_x0_y0 loc_x0_y1 loc_x0_y2 loc_x0_y3 loc_x0_y4 loc_x0_y5 loc_x0_y6 loc_x0_y7 loc_x1_y0 loc_x1_y1 loc_x1_y2 loc_x1_y3 loc_x1_y4 loc_x1_y5 loc_x1_y6 loc_x1_y7 - place)
    (:init (at-robot loc_x0_y4) (connected loc_x0_y0 loc_x0_y1) (connected loc_x0_y0 loc_x1_y0) (connected loc_x0_y1 loc_x0_y0) (connected loc_x0_y1 loc_x0_y2) (connected loc_x0_y1 loc_x1_y1) (connected loc_x0_y2 loc_x0_y1) (connected loc_x0_y2 loc_x0_y3) (connected loc_x0_y2 loc_x1_y2) (connected loc_x0_y3 loc_x0_y2) (connected loc_x0_y3 loc_x0_y4) (connected loc_x0_y3 loc_x1_y3) (connected loc_x0_y4 loc_x0_y3) (connected loc_x0_y4 loc_x0_y5) (connected loc_x0_y4 loc_x1_y4) (connected loc_x0_y5 loc_x0_y4) (connected loc_x0_y5 loc_x0_y6) (connected loc_x0_y5 loc_x1_y5) (connected loc_x0_y6 loc_x0_y5) (connected loc_x0_y6 loc_x0_y7) (connected loc_x0_y6 loc_x1_y6) (connected loc_x0_y7 loc_x0_y6) (connected loc_x0_y7 loc_x1_y7) (connected loc_x1_y0 loc_x0_y0) (connected loc_x1_y0 loc_x1_y1) (connected loc_x1_y1 loc_x0_y1) (connected loc_x1_y1 loc_x1_y0) (connected loc_x1_y1 loc_x1_y2) (connected loc_x1_y2 loc_x0_y2) (connected loc_x1_y2 loc_x1_y1) (connected loc_x1_y2 loc_x1_y3) (connected loc_x1_y3 loc_x0_y3) (connected loc_x1_y3 loc_x1_y2) (connected loc_x1_y3 loc_x1_y4) (connected loc_x1_y4 loc_x0_y4) (connected loc_x1_y4 loc_x1_y3) (connected loc_x1_y4 loc_x1_y5) (connected loc_x1_y5 loc_x0_y5) (connected loc_x1_y5 loc_x1_y4) (connected loc_x1_y5 loc_x1_y6) (connected loc_x1_y6 loc_x0_y6) (connected loc_x1_y6 loc_x1_y5) (connected loc_x1_y6 loc_x1_y7) (connected loc_x1_y7 loc_x0_y7) (connected loc_x1_y7 loc_x1_y6) (visited loc_x0_y4))
    (:goal (and (visited loc_x0_y0) (visited loc_x0_y1) (visited loc_x0_y2) (visited loc_x0_y3) (visited loc_x0_y4) (visited loc_x0_y5) (visited loc_x0_y6) (visited loc_x0_y7) (visited loc_x1_y0) (visited loc_x1_y1) (visited loc_x1_y2) (visited loc_x1_y3) (visited loc_x1_y4) (visited loc_x1_y5) (visited loc_x1_y6) (visited loc_x1_y7)))
)