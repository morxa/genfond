(define (problem blocks-010-1)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b1) (clear b6) (clear b8) (handempty) (on b2 b9) (on b3 b5) (on b4 b3) (on b5 b7) (on b6 b2) (on b8 b4) (ontable b0) (ontable b1) (ontable b7) (ontable b9))
    (:goal (and (clear b9) (on b9 b5) (on b5 b1) (on b1 b8) (on b8 b6) (on b6 b4) (on b4 b2) (on b2 b0) (on b0 b7) (ontable b7)))
)