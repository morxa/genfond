(define (problem blocks-008-2)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7)
    (:init (clear b0) (clear b2) (clear b6) (handempty) (on b1 b5) (on b2 b1) (on b3 b7) (on b4 b3) (on b5 b4) (ontable b0) (ontable b6) (ontable b7))
    (:goal (and (on b3 b0) (on b0 b7) (on b7 b6) (on b6 b4) (on b4 b1) (on b1 b2)))
)