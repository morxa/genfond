(define (problem problem-2-5)
    (:domain miconic)
    (:objects f0 f1 f2 f3 - floor p0 p1 - passenger)
    (:init (above f0 f1) (above f0 f2) (above f0 f3) (above f1 f2) (above f1 f3) (above f2 f3) (destin p0 f2) (destin p1 f0) (lift-at f0) (origin p0 f1) (origin p1 f1))
    (:goal (and (served p0) (served p1)))
)