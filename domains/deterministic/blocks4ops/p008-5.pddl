(define (problem blocks-008-5)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7)
    (:init (clear b1) (clear b4) (clear b6) (handempty) (on b0 b5) (on b1 b7) (on b2 b0) (on b4 b2) (on b7 b3) (ontable b3) (ontable b5) (ontable b6))
    (:goal (and (on b6 b4) (on b4 b3) (on b3 b2) (on b2 b5) (on b5 b1) (on b1 b0)))
)