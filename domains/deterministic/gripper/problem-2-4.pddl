(define (problem gripper-2-4)
    (:domain gripper-strips)
    (:objects ball1 ball2 left right rooma roomb)
    (:init (at ball1 rooma) (at ball2 rooma) (at-robby rooma) (ball ball1) (ball ball2) (free left) (free right) (gripper left) (gripper right) (room rooma) (room roomb))
    (:goal (and (at ball1 roomb) (at ball2 roomb)))
)