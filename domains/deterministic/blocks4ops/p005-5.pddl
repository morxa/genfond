(define (problem blocks-005-5)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b3) (handempty) (on b0 b4) (on b1 b0) (on b3 b1) (on b4 b2) (ontable b2))
    (:goal (and (on b2 b3) (on b3 b4) (on b4 b1)))
)