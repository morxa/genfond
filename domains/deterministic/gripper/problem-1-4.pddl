(define (problem gripper-1-4)
    (:domain gripper-strips)
    (:objects ball1 left right rooma roomb)
    (:init (at ball1 rooma) (at-robby rooma) (ball ball1) (free left) (free right) (gripper left) (gripper right) (room rooma) (room roomb))
    (:goal (at ball1 roomb))
)