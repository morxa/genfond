(define (problem blocks-005-4)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b1) (clear b4) (handempty) (on b1 b0) (on b2 b3) (on b4 b2) (ontable b0) (ontable b3))
    (:goal (clear b0))
)