(define (problem blocks-010-01)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b1) (clear b5) (clear b8) (clear b9) (handempty) (on b2 b7) (on b5 b6) (on b6 b2) (on b7 b3) (on b8 b4) (on b9 b0) (ontable b0) (ontable b1) (ontable b3) (ontable b4))
    (:goal (on b6 b2))
)