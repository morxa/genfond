(define (problem blocks-005-4)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b3) (clear b4) (handempty) (on b1 b2) (on b2 b0) (on b4 b1) (ontable b0) (ontable b3))
    (:goal (and (on b2 b4) (on b4 b0)))
)